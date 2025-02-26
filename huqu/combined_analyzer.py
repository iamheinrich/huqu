from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Any
import pandas as pd
from IPython.display import display, HTML

@dataclass
class AnalyzerConfig:
    """Configuration for analysis parameters."""
    over_threshold: float = 0.5
    under_threshold: float = 0.1
    figure_size: Tuple[int, int] = (12, 6)
    flags: Dict[str, str] = field(default_factory=lambda: {
        "over": "⚠️ Overrepresented",
        "under": "🚨 Underrepresented",
        "missing": "❗ Missing"
    })
    rare_threshold: int = 3  # Threshold for rare attributes


class BaseAnalyzer:
    """Base class for data distribution analyzers."""
    
    REQUIRED_COLUMNS = {'class', 'dimension', 'attribute', 'image_id'}
    
    def __init__(self, data_dict: Optional[Dict[str, pd.DataFrame]] = None, 
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize analyzer with data splits and optional configuration.
        
        Args:
            data_dict: Dictionary mapping split names to DataFrames
            config: Optional configuration parameters
        """
        self.config = AnalyzerConfig(**(config or {}))
        self.data_dict = None
        
        if data_dict is not None:
            self.load_data(data_dict)
    
    def load_data(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """
        Load and validate the input data.
        
        Args:
            data_dict: Dictionary mapping split names to DataFrames
        """
        self._validate_input(data_dict)
        self.data_dict = data_dict
        
        # Extract common metadata
        combined_df = pd.concat(data_dict.values(), ignore_index=True)
        self.unique_classes = sorted(combined_df['class'].unique())
        self.unique_dimensions = sorted(combined_df['dimension'].unique())
        self.attributes_by_dimension = {
            dim: sorted(combined_df.loc[combined_df['dimension'] == dim, 'attribute'].unique())
            for dim in self.unique_dimensions
        }
    
    def _validate_input(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """
        Validate required columns exist in all data splits.
        
        Args:
            data_dict: Dictionary mapping split names to DataFrames
        
        Raises:
            ValueError: If required columns are missing
        """
        missing_cols = {
            split_name: self.REQUIRED_COLUMNS - set(df.columns) 
            for split_name, df in data_dict.items() 
            if self.REQUIRED_COLUMNS - set(df.columns)
        }
        
        if missing_cols:
            error_msg = "\n".join(f"Split '{name}' missing columns: {cols}" 
                                 for name, cols in missing_cols.items())
            raise ValueError(f"Data validation failed: {error_msg}")
    
    def _print_section(self, title: str, data: Any) -> None:
        """
        Format and print a section of the report.
        
        Args:
            title: Section title
            data: Data to display (DataFrame or list)
        """
        print(f"\n{'-' * 30}\n{title}\n{'-' * 30}")
        
        if isinstance(data, list):
            for line in data:
                print(line)
        elif isinstance(data, pd.DataFrame):
            display(HTML(data.to_html(
                na_rep=self.config.flags["missing"], 
                float_format=lambda x: f"{int(x)}" if pd.notnull(x) else ""
            )))
        else:
            print(data)
