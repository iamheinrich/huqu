from typing import Any, Dict
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from ..prompts.templates import PromptTemplates
from .base import PipelineStage
from ..models.base import BaseModel

class CaptionGenerationStage(PipelineStage):
    """Generates captions for images using a multimodal model."""
    
    def __init__(self, model: BaseModel):
        """Initialize with a single multimodal model.
        
        Args:
            model: Model implementing image-to-text generation interface
        """
        super().__init__(models={"mllm": model})
        self.batch_size = self.config["stages"]["caption_generation"]["batch_size"]
        self.class_name = self.config["dataset"]["class_name"]
    def _generate_single_caption(self, image: Any, prompt: str) -> str:
        """Generate a caption for a single image with basic error handling.
        
        Args:
            image: The image to caption
            prompt: The prompt to use for caption generation
            
        Returns:
            Generated caption or error message
        """
        
        try:
            return self.models["mllm"].generate(image, prompt)
        except Exception as e:
            return f"[Error: {str(e)}]"
    
    def process(self, dataset: Dataset) -> pd.DataFrame:
        """Generate captions for all images in the dataset.
        
        Args:
            dataset: Hugging Face dataset containing images
            
        Returns:
            DataFrame with image_ids and captions
        """
        image_ids = []
        captions = []
        
        
        for idx in tqdm(range(len(dataset)), desc="Generating captions"):
            # Get image and label from dataset
            datapoint = dataset[idx]
            class_name = self.class_name
            
            # Generate caption
            prompt = PromptTemplates.image_caption(class_name)
            caption = self._generate_single_caption(datapoint["image"], prompt)
            
            # Store results
            image_ids.append(datapoint["image_id"])
            captions.append(caption)
        
        # Create DataFrame with results
        df = pd.DataFrame({
            'image_id': image_ids,
            'caption': captions
        })
        
        # Save DataFrame if path is configured
        df_path = self.config["dataset"]["captions_path"]
        if df_path:
            df.to_parquet(
                df_path,
                compression=self.config["dataset"]["compression"]
            )
        
        return df
