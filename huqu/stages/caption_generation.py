from typing import Any
from tqdm import tqdm
from datasets import Dataset
from ..prompts.templates import PromptTemplates


class CaptionGenerationStage:
    """Generates captions for images using a multimodal model."""
    
    def __init__(self, model: Any):
        """Initialize with a multimodal model.
        
        Args:
            model: Model that can generate captions from images
        """
        self.model = model
    
    def _generate_single_caption(self, image: Any, prompt: str) -> str:
        """Generate a caption for a single image with basic error handling.
        
        Args:
            image: The image to caption
            prompt: The prompt to use for caption generation
            
        Returns:
            Generated caption or error message
        """
        try:
            return self.model.generate(image, prompt)
        except Exception as e:
            return f"[Error: Caption generation failed - {str(e)}]"
    
    def process(self, dataset: Dataset) -> Dataset:
        """Generate captions for all images in the dataset.
        
        Args:
            dataset: Hugging Face dataset containing images
            
        Returns:
            Dataset with added 'caption' column
        """
        captions = []
        
        for idx in tqdm(range(len(dataset)), desc="Generating captions"):
            example = dataset[idx]
            
            # Get label if available
            label = str(example["label"]) if "label" in example else None
            
            # Generate caption with appropriate prompt
            prompt = PromptTemplates.image_caption(label) if label else PromptTemplates.image_caption()
            caption = self._generate_single_caption(example["image"], prompt)
            captions.append(caption)
        
        # Add captions as new column
        return dataset.add_column("caption", captions)
