from typing import Any, List, Dict, Set, Optional
import pandas as pd
from tqdm import tqdm
from .base import PipelineStage
from ..prompts.templates import PromptTemplates
from ..models.base import BaseModel
from ..types import ClassificationCriteria

#TODO: We should think about adding config file to define the most important params like dataframe, num_refining_rounds, sample_size, etc.
#TODO: We need a well-defined interface for storing the dimensions and attributes

class CriteriaInitializationStage(PipelineStage):
    """Discovers and refines dimensions and attributes for categorizing images based on their captions.
    
    This stage implements a two-pass approach:
    1. First Pass: Process all batches to collect and summarize dimensions
    2. Second Pass: Using summarized dimensions, collect and summarize attributes
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
        """Combine summarization and cleanup of dimension keywords in one call."""
        dims_list = list(all_dimensions)
        prompt = PromptTemplates.summarize_and_reduce_dimensions(dims_list)
        print(f"\n[SUMMARIZE & REDUCE DIMENSIONS] Prompt:\n{prompt}")
        response = self.models["llm"].generate(prompt)
        print(f"[SUMMARIZE & REDUCE DIMENSIONS] Response:\n{response}")

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
            print(f"\n[COLLECT ATTRIBUTES] Dimension: {dimension}")
            print(f"[COLLECT ATTRIBUTES] Prompt:\n{feature_prompt}")
            feature_response = self.models["llm"].generate(feature_prompt)
            print(f"[COLLECT ATTRIBUTES] Response:\n{feature_response}")
            
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
                print(f"\n[SUMMARIZE & REDUCE ATTRIBUTES] Dimension: {dimension}")
                print(f"[SUMMARIZE & REDUCE ATTRIBUTES] Prompt:\n{prompt}")
                response = self.models["llm"].generate(prompt)
                print(f"[SUMMARIZE & REDUCE ATTRIBUTES] Response:\n{response}")

                search_key = "Final Attributes:"
                if search_key in response:
                    after_key = response.split(search_key, 1)[1].strip()
                    final_attrs = {a.strip().lower() for a in after_key.split(",") if a.strip()}
                    final_attributes[dimension] = final_attrs
                else:
                    # fallback
                    final_attributes[dimension] = set(attrs_list)
            else:
                final_attributes[dimension] = set()

        self.dimension_attributes = final_attributes
        
    def process(self) -> ClassificationCriteria:
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
        
        return ClassificationCriteria(
            dimensions=self.summarized_dimensions,
            attributes={
                dim: list(attrs) for dim, attrs in self.dimension_attributes.items()
            }
        )
