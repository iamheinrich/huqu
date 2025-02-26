from typing import Any, List, Dict, Set, Optional, TypedDict
import pandas as pd
from tqdm import tqdm
import logging
from .base import PipelineStage
from ..prompts.templates import PromptTemplates
from ..models.base import BaseModel
import json

logger = logging.getLogger(__name__)

class ClassificationCriteria(TypedDict):
    """Type definition for classification criteria.
    
    Structure:
        dimensions: List of dimension keywords
        attributes: Dictionary mapping each dimension to its list of attributes
    """
    dimensions: List[str]
    attributes: Dict[str, List[str]]

class CriteriaInitializationStage(PipelineStage):
    """Pipeline stage for discovering and initializing classification criteria.
    
    This stage analyzes image captions to identify meaningful dimensions and attributes
    for categorizing images. It uses a two-pass approach:
    1. First Pass: Process all batches to collect and summarize dimensions
    2. Second Pass: Using summarized dimensions, collect and summarize attributes
    
    Configuration:
        The stage requires the following config parameters:
        - dataset.captions_path: Path to the parquet file containing captions
        - dataset.unrefined_criteria_path: Path to save the initial criteria
        - stages.criteria_init.batch_size: Number of captions to process per batch
        - dataset.main_subject: The main subject being analyzed (e.g., "cat")
    """
    
    def __init__(self, model: BaseModel):
        """Initialize the criteria initialization stage.
        
        Args:
            model: Language model for analyzing captions
        """
        super().__init__(models={"llm": model})
        self.batch_size = self.config["stages"]["criteria_init"]["batch_size"]
        self.main_subject = self.config["dataset"]["main_subject"]
        self.summarized_dimensions: List[str] = []  # Final set of dimensions after summarization
        self.dimension_attributes: Dict[str, Set[str]] = {}  # Attributes per dimension
        
    
    def _discover_dimensions_from_batch(self, batch_captions: List[str]) -> Set[str]:
        """Discover dimensions from a single batch of captions.
        
        Args:
            batch_captions: List of captions to analyze
        Returns:
            Set of discovered dimensions
        """
        # Sanitize captions first
        sanitized_captions = [c.replace("\n", ", ") for c in batch_captions]
        dimension_prompt = PromptTemplates.get_dimensions(
            main_subject=self.main_subject,
            batch_size=len(batch_captions),
            caption_samples="\n".join(sanitized_captions)  
        )
        dimension_response = self.models["llm"].generate(dimension_prompt)
        dimensions = set()
        if "Suggested Dimension:" in dimension_response:
            dims = dimension_response.split("Suggested Dimension:")[1].strip()
            dimensions = {d.strip().lower() for d in dims.split(",")}
        
        return dimensions
    
    
    def _summarize_and_reduce_dimensions(self, all_dimensions: Set[str]) -> List[str]:
        """Summarize and clean up dimension keywords to create a final set.
        
        Takes the raw set of dimensions discovered from all batches and:
        1. Removes duplicates and near-duplicates
        2. Standardizes formatting and terminology
        3. Combines related dimensions
        4. Ensures dimensions are meaningful for the main subject
        
        Args:
            all_dimensions: Set of all raw dimensions discovered from captions
            
        Returns:
            List of cleaned and standardized dimension keywords
        """
        dims_list = list(all_dimensions)
        prompt = PromptTemplates.summarize_and_reduce_dimensions(dims_list)
        response = self.models["llm"].generate(prompt)

        # Try to parse "Final Dimensions: ..." from response
        search_key = "Final Dimensions:"
        if search_key in response:
            after_key = response.split(search_key, 1)[1].strip()
            final_dims = [d.strip().lower() for d in after_key.split(",") if d.strip()]
            return final_dims

        # Fallback if we can't parse
        return list(all_dimensions)

    
    def _collect_attributes_from_batch(self, batch_captions: List[str]) -> Dict[str, Set[str]]:
        """Collect attributes for each summarized dimension from a batch.
        
        Args:
            batch_captions: List of captions to analyze
            
        Returns:
            Dictionary mapping dimensions to sets of attributes
        """
        # Sanitize captions first
        sanitized_captions = [c.replace("\n", ", ") for c in batch_captions]
        batch_results: Dict[str, Set[str]] = {}
        
        for dimension in self.summarized_dimensions:
            feature_prompt = PromptTemplates.get_features(
                main_subject=self.main_subject,
                dimension=dimension,
                caption_samples="\n".join(sanitized_captions)  # Safe join
            )
            feature_response = self.models["llm"].generate(feature_prompt)
            
            response_lower = feature_response.lower()
            if "dimension:" in response_lower:
                feats = response_lower.split("dimension:")[1].strip()
                attributes = {f.strip().lower() for f in feats.split(",")}
                batch_results[dimension] = attributes
            elif f"{dimension}" in response_lower:
                feats = response_lower.split(f"{dimension}")[1].strip()
                attributes = {f.strip().lower() for f in feats.split(",")}
                batch_results[dimension] = attributes
        
        return batch_results
    
    def _summarize_and_reduce_attributes(self) -> None:
        """Combine summarization and reduction of attributes for each dimension."""
        final_attributes: Dict[str, Set[str]] = {}

        for dimension in self.summarized_dimensions:
            if dimension in self.dimension_attributes:
                attrs_list = list(self.dimension_attributes[dimension])
                prompt = PromptTemplates.summarize_and_reduce_attributes(dimension, attrs_list)
                response = self.models["llm"].generate(prompt)

                search_key = "Final Attributes:"
                if search_key in response:
                    after_key = response.split(search_key, 1)[1].strip()
                    final_attrs = {a.strip().lower() for a in after_key.split(",") if a.strip()}
                    final_attributes[dimension] = final_attrs
                else:
                    # fallback
                    logger.warning(f"Could not parse attributes for dimension {dimension}, using original list")
                    final_attributes[dimension] = set(attrs_list)
            else:
                final_attributes[dimension] = set()

        self.dimension_attributes = final_attributes
        
    def process(self) -> ClassificationCriteria:
        """Process captions to discover and initialize classification criteria.
        
        Loads captions from the configured captions_path, processes them to discover
        dimensions and attributes, and saves the initial criteria to the configured
        unrefined_criteria_path as JSON.
        
        Returns:
            Dict containing the discovered dimensions and attributes with structure:
            {
                "dimensions": List[str],  # List of dimension keywords
                "attributes": Dict[str, List[str]]  # Dimension -> attributes mapping
            }
            Note: While this method returns the criteria, it also saves them to the
            configured path for use by subsequent stages.
        """
        df_path = self.config["dataset"]["captions_path"]
        df = pd.read_parquet(df_path)
        
        # 1) Collect & unify discovered dimensions from each batch
        all_dimensions: Set[str] = set()
        for i in tqdm(range(0, len(df), self.batch_size), desc="Discovering dimensions"):
            batch_captions = df['caption'].iloc[i:i + self.batch_size].tolist()
            batch_dims = self._discover_dimensions_from_batch(batch_captions)
            all_dimensions.update(batch_dims)
        
        # 2) Summarize & reduce dimensions in one call
        self.summarized_dimensions = self._summarize_and_reduce_dimensions(all_dimensions)
        
        # 3) Collect attributes for these final dimensions
        for i in tqdm(range(0, len(df), self.batch_size), desc="Collecting attributes"):
            batch_captions = df['caption'].iloc[i:i + self.batch_size].tolist()
            batch_attrs = self._collect_attributes_from_batch(batch_captions)
            for dim, attrs in batch_attrs.items():
                if dim not in self.dimension_attributes:
                    self.dimension_attributes[dim] = set()
                self.dimension_attributes[dim].update(attrs)
        
        # 4) Summarize & reduce attributes for each dimension in one call
        self._summarize_and_reduce_attributes()
        
        # 5) Filter out dimensions that lost all attributes
        self.summarized_dimensions = [
            dim for dim in self.summarized_dimensions 
            if self.dimension_attributes.get(dim, set())
        ]
        
        # Build criteria dictionary with type checking
        criteria: ClassificationCriteria = {
            "dimensions": self.summarized_dimensions,
            "attributes": {
                dim: list(attrs) for dim, attrs in self.dimension_attributes.items()
            }
        }

        # Save criteria using config path
        output_path = self.config["dataset"]["unrefined_criteria_path"]
        with open(output_path, "w") as f:
            json.dump(criteria, f, indent=2)
        
        return criteria
