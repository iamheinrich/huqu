from typing import Any, Dict, List
import pandas as pd
from tqdm import tqdm
import logging
from .base import PipelineStage
from ..models.base import BaseModel
from ..prompts.templates import PromptTemplates
import json

logger = logging.getLogger(__name__)

class ImageAssignmentStage(PipelineStage):
    """Pipeline stage for assigning images to attributes within each dimension.
    
    This stage:
    1. Loads captions from the configured captions_path
    2. Loads refined criteria from the configured refined_criteria_path
    3. Assigns each image to attributes for each dimension using the classification prompt
    4. Builds a Semantic Structure Description (SSD) DataFrame
    5. Saves the SSD to the configured assignments_path
    """
    
    def __init__(self, model: BaseModel):
        """Initialize the image assignment stage.
        
        Args:
            model: Language model for classification
        """
        super().__init__(models={"llm": model})
        self.class_name = self.config["dataset"]["class_name"]
    
    def process(self, **kwargs) -> None:
        """Process captions and assign them to attributes for each dimension.
        
        Loads captions and refined criteria from their configured paths,
        processes each caption to assign attributes for each dimension,
        builds a Semantic Structure Description (SSD) DataFrame, and
        saves it to the configured assignments_path.
        
        The SSD DataFrame contains columns:
        - class: The class name of the images
        - dimension: The dimension being classified
        - attribute: The assigned attribute for that dimension
        - image_id: The ID of the image being classified
        """
        # Load captions and refined criteria
        df = pd.read_parquet(self.config["dataset"]["captions_path"])
        with open(self.config["dataset"]["refined_criteria_path"]) as f:
            criteria = json.load(f)
        
        # Ensure required columns exist
        required_cols = {"caption", "image_id"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame is missing required column(s): {', '.join(missing)}"
            )

        # Get dimensions and their attributes
        dimensions = {
            dim: criteria["attributes"][dim]
            for dim in criteria["dimensions"]
        }
        
        # Process each caption for each dimension
        for dimension, attributes in tqdm(dimensions.items(), desc="Processing dimensions"):
            # We assume here that attributes is a non-empty list
            assignments = []
            for caption in tqdm(df["caption"], desc=f"Processing {dimension}", leave=False):
                assigned = self._assign_attribute(caption, dimension, attributes)
                assignments.append(assigned)
            df[dimension] = assignments
        
        # Build the final SSD
        final_ssd = pd.DataFrame(columns=["class", "dimension", "attribute", "image_id"])
        for idx, row in df.iterrows():
            for dimension in dimensions.keys():
                final_ssd.loc[len(final_ssd)] = {
                    "class": self.class_name,
                    "dimension": dimension,
                    "attribute": row[dimension],
                    "image_id": row["image_id"]
                }
                
        # Save results
        output_path = self.config["dataset"]["assignments_path"]
        final_ssd.to_parquet(
            output_path,
            compression=self.config["dataset"]["compression"]
        )
        logger.info(f"Saved SSD DataFrame to {output_path}")
    
    def _assign_attribute(self, caption: str, dimension: str, attributes: List[str]) -> str:
        """
        Assign a single caption to an attribute for a given dimension using
        the final classification prompt.
        
        Args:
            caption: The text caption to classify
            dimension: The dimension we are classifying for
            attributes: A list of valid attributes for this dimension
        
        Returns:
            The chosen attribute (a string).
        """
        # Build the prompt for assignment
        features_str = ", ".join(attributes)
        prompt = PromptTemplates.assign_attribute(
            dimension=dimension,
            caption=caption,
            features=features_str
        )
        
        # Generate classification
        raw_response = self.models["llm"].generate(prompt)
        # Strip braces/spaces
        response = raw_response.strip().strip("{}").strip()
        
        # If the model's response matches an attribute (case-insensitive), use it
        if response.lower() in [attr.lower() for attr in attributes]:
            return response
        
        # Otherwise, fallback to the first attribute
        logger.debug(f"No matching attribute found for {dimension}, using default")
        return attributes[0]
