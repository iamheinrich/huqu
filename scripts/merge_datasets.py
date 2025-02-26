from pathlib import Path
import pandas as pd
from typing import Tuple

def load_and_split_data(file_path: str, num_training_images: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load assignment data and split into train/test based on dimensions.
    
    Args:
        file_path: Path to the parquet file containing assignments
        num_training_images: Number of training images per dimension
        
    Returns:
        Tuple of (train_df, test_df)
    """
    df = pd.read_parquet(file_path)
    n_dims = df['dimension'].nunique()
    n_train = num_training_images * n_dims
    
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]
    return train_df, test_df

def main():
    num_training_images = 2  # Can be modified as needed
    
    # File paths
    data_dir = Path("data")
    cats_file = data_dir / "assignments_cats.parquet"
    dogs_file = data_dir / "assignments_dogs.parquet"
    
    # Load and split data
    cats_train, cats_test = load_and_split_data(str(cats_file), num_training_images)
    dogs_train, dogs_test = load_and_split_data(str(dogs_file), num_training_images)
    
    # Merge datasets
    train_df = pd.concat([cats_train, dogs_train])
    test_df = pd.concat([cats_test, dogs_test])
    
    # Save merged datasets
    train_df.to_parquet(data_dir / "merged_train.parquet")
    test_df.to_parquet(data_dir / "merged_test.parquet")

if __name__ == "__main__":
    main() 