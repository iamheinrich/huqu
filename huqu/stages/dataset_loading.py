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
        self.dataset_name = self.config["dataset"]["name"]
        self.main_subject = self.config["dataset"]["main_subject"]
        self.split = self.config["dataset"]["split"]
        self.config_name = self.config["dataset"]["config_name"]
        
        # Load dataset
        #TODO: Automate filtering by labels.
        dataset_train = load_dataset(self.dataset_name, split="train")
        dataset_test = load_dataset(self.dataset_name, split="test")
        dataset_dogs_train = dataset_train.filter(lambda record: record["labels"] == 1)
        dataset_dogs_test = dataset_test.filter(lambda record: record["labels"] == 1)
        dataset_dogs_train = dataset_dogs_train.select(range(100))
        dataset_dogs_test = dataset_dogs_test.select(range(50))
        dataset = concatenate_datasets([dataset_dogs_train, dataset_dogs_test])
        # Add image_id if not present
        if "image_id" not in dataset.features:
            dataset = dataset.add_column("image_id", [f"img_{i+150}" for i in range(len(dataset))])
        
        return dataset 