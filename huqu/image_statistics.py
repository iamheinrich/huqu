from PIL import Image
import numpy as np
from typing import Dict
from datasets import DatasetDict
from tqdm.auto import tqdm
import warnings
import pandas as pd

class DatasetAnalyzer:
    """A class to analyze image statistics in Hugging Face datasets."""
    
    def __init__(self, hf_dataset: DatasetDict, image_key: str = "image"):
        """
        Initialize the dataset analyzer.
        
        Args:
            hf_dataset: Hugging Face dataset dictionary
            image_key: Key containing the images in the dataset
        """
        self.dataset = hf_dataset
        self.image_key = image_key
        self.stats = {}
        self._validate_dataset()
        
    def _validate_dataset(self) -> None:
        """Validate that the dataset has the expected structure."""
        if not isinstance(self.dataset, DatasetDict):
            raise ValueError("Dataset must be a DatasetDict")
            
        # Check that we have at least one split
        if not self.dataset:
            raise ValueError("Dataset is empty")
            
        # Check the first item of the first split to validate structure
        first_split = next(iter(self.dataset.values()))
        if len(first_split) == 0:
            raise ValueError("Dataset split is empty")
            
        first_item = first_split[0]
        if not isinstance(first_item, dict) or self.image_key not in first_item:
            raise ValueError(f"Dataset items must be dictionaries with an '{self.image_key}' key")
            
        # Check that the image is a PIL Image
        if not isinstance(first_item[self.image_key], Image.Image):
            raise ValueError(f"The '{self.image_key}' field must contain PIL Image objects")
    
    def analyze(self) -> None:
        """Analyze all splits in the dataset."""
        for split in self.dataset:
            print(f"\nAnalyzing {split} split...")
            self.stats[split] = self._analyze_split(split)
    
    def _analyze_split(self, split: str) -> Dict:
        """Analyze a specific split of the dataset."""
        split_data = self.dataset[split]
        total_images = len(split_data)
        valid_stats = []
        
        # Process each image individually
        for item in tqdm(split_data, desc=f"Computing statistics for {total_images} images"):
            if isinstance(item, dict) and self.image_key in item:
                stats = self._process_image(item[self.image_key])
                if stats is not None:
                    valid_stats.append(stats)
        
        if not valid_stats:
            raise ValueError(f"No valid images found in {split} split")
            
        # Aggregate statistics
        n_images = len(valid_stats)
        stats_keys = valid_stats[0].keys()
        
        # Calculate means for all metrics
        aggregated_stats = {
            'n_images': n_images
        }
        
        for key in stats_keys:
            if key != 'area':  # Area will be reported as mean
                aggregated_stats[f'mean_{key}'] = sum(s[key] for s in valid_stats) / n_images
        
        return aggregated_stats
    
    def _process_image(self, image_data) -> Dict:
        """Process a single image and return its statistics."""
        try:
            if not isinstance(image_data, Image.Image):
                return None
                
            # Convert to grayscale for brightness and contrast calculation
            gray_img = image_data.convert('L')
            pixels = np.array(gray_img)
            
            # Get RGB channels
            rgb_img = np.array(image_data.convert('RGB'))
            r, g, b = rgb_img[:, :, 0], rgb_img[:, :, 1], rgb_img[:, :, 2]
            
            # Calculate relative intensities
            rgb_sum = r.astype(float) + g.astype(float) + b.astype(float)
            # Avoid division by zero
            rgb_sum[rgb_sum == 0] = 1
            
            r_relative = r.astype(float) / rgb_sum
            g_relative = g.astype(float) / rgb_sum
            b_relative = b.astype(float) / rgb_sum
            
            return {
                'brightness': float(np.mean(pixels)),
                'rms_contrast': float(np.std(pixels)),
                'red_relative_intensity': float(np.mean(r_relative)),
                'green_relative_intensity': float(np.mean(g_relative)),
                'blue_relative_intensity': float(np.mean(b_relative)),
                'area': float(pixels.shape[0] * pixels.shape[1])
            }
            
        except Exception as e:
            warnings.warn(f"Error processing image: {str(e)}")
            return None
    
    def report(self) -> None:
        """Generate and print a report of the dataset statistics in tabular format using pandas."""
        if not self.stats:
            self.analyze()
            
        # Create data for each split
        data = {}
        for split in self.stats:
            data[split] = {
                'n_images': f"{self.stats[split]['n_images']:,d}",
                'mean_brightness': f"{self.stats[split]['mean_brightness']:.2f}",
                'mean_rms_contrast': f"{self.stats[split]['mean_rms_contrast']:.2f}",
                'mean_red_relative_intensity': f"{self.stats[split]['mean_red_relative_intensity']:.3f}",
                'mean_green_relative_intensity': f"{self.stats[split]['mean_green_relative_intensity']:.3f}",
                'mean_blue_relative_intensity': f"{self.stats[split]['mean_blue_relative_intensity']:.3f}",
                'mean_area': f"{self.stats[split]['mean_area']:,.0f}"
            }
        
        # Create DataFrame with metrics as index and splits as columns
        df = pd.DataFrame(data)
        
        print("\nDataset Statistics Report")
        print("=" * 50)
        print(df.to_string()) 