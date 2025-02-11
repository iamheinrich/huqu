from typing import Any, Dict, List
import pandas as pd
from tqdm import tqdm
from .base import PipelineStage
from ..models.base import BaseModel
from ..prompts.templates import PromptTemplates


class ImageAssignmentStage(PipelineStage):
    """Pipeline stage for assigning images to attributes within each dimension
    using the final classification prompt.
    """
    
    def __init__(self, model: BaseModel):
        """Initialize the image assignment stage.
        
        Args:
            model: Language model for classification
        """
        super().__init__(models={"llm": model})
        self.class_name = self.config["dataset"]["class_name"]
    
    def process(self, criteria: Dict[str, Any], **kwargs) -> pd.DataFrame:
        """
        Process captions and assign them to attributes for each dimension.
        
        Args:
            criteria: Dictionary containing dimensions and their attributes
        
        Returns:
            DataFrame with captions and their assigned attributes
        """
        # Load captions
        df = pd.read_parquet(self.config["dataset"]["captions_path"])
        
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
                
        # Optionally save results
        output_path = self.config["dataset"]["assignments_path"]
        if output_path:
            final_ssd.to_parquet(
                output_path,
                compression=self.config["dataset"]["compression"]
            )
        
        return final_ssd
    
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
        return attributes[0]
