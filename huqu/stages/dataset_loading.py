from datasets import load_dataset, Dataset, concatenate_datasets
from typing import Optional
from .base import PipelineStage
import pandas as pd

class DatasetLoadingStage(PipelineStage):
    """Loads datasets from Hugging Face Hub."""
    def __init__(self):
        super().__init__()
    
    def process(
        self,
        **kwargs
    ) -> Dataset:
        """Load a dataset from Hugging Face Hub.
        
        Args:
            **kwargs: Additional arguments passed to load_dataset
            
        Returns:
            Hugging Face Dataset with guaranteed image_id column and main_subject in metadata
        """
        # Only keep self attributes that are used
        self.dataset_path = self.config["dataset"]["path"]
        self.dataset_config_name = self.config["dataset"]["config_name"]
        self.class_label = self.config["dataset"].get("class_label")
        self.num_train_samples = self.config["dataset"].get("num_train_samples") 
        self.num_test_samples = self.config["dataset"].get("num_test_samples")
        
        # Load and filter train split
        train_ds = load_dataset(self.dataset_path, self.dataset_config_name, split="train")
        train_ds = train_ds.filter(lambda x: x["labels"] == self.class_label)
        train_ds = train_ds.select(range(self.num_train_samples))
        
        # Load and filter test split 
        test_ds = load_dataset(self.dataset_path, self.dataset_config_name, split="test")
        test_ds = test_ds.filter(lambda x: x["labels"] == self.class_label)
        test_ds = test_ds.select(range(self.num_test_samples))
        
        # Concatenate train and test splits
        dataset = concatenate_datasets([train_ds, test_ds])
        
        # Add image_id if not present
        if "image_id" not in dataset.features:
            dataset = dataset.add_column("image_id", [i for i in range(len(dataset))])
        
        return dataset