from datasets import load_dataset, Dataset
from typing import Optional


class DatasetLoadingStage:
    """Loads datasets from Hugging Face Hub."""
    
    def process(
        self,
        dataset_name: str,
        main_subject: str,
        split: str = "test",
        **kwargs
    ) -> Dataset:
        """Load a dataset from Hugging Face Hub.
        
        Args:
            dataset_name: Name of the dataset on Hugging Face Hub
            main_subject: Main subject/theme of the dataset (e.g., "animals", "vehicles")
            split: Dataset split to load (default: "test")
            **kwargs: Additional arguments passed to load_dataset
            
        Returns:
            Hugging Face Dataset with guaranteed image_id column and main_subject in metadata
        """
        # Load dataset
        dataset = load_dataset(dataset_name, split=split, **kwargs)
        
        # Add image_id if not present
        if "image_id" not in dataset.features:
            dataset = dataset.add_column("image_id", [f"img_{i}" for i in range(len(dataset))])
        
        # Store main_subject in dataset metadata
        original_description = dataset.info.description or ""
        dataset.info.description = f"{original_description}\nMain subject: {main_subject}"
        
        return dataset 