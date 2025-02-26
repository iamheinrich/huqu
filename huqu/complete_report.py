from typing import Dict, Optional, Any
import pandas as pd

from huqu.interclass_analysis import InterClassAnalyzer
from huqu.intraclass_analysis import IntraClassAnalyzer
from huqu.combined_analyzer import AnalyzerConfig


class DataAnalyzer:
    """Unified interface for data distribution analysis."""
    
    def __init__(self, data_dict: Optional[Dict[str, pd.DataFrame]] = None, 
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize both types of analyzers with shared data and configuration.
        
        Args:
            data_dict: Dictionary mapping split names to DataFrames
            config: Optional configuration parameters
        """
        self.config = AnalyzerConfig(**(config or {}))
        self.intra = IntraClassAnalyzer(data_dict, config)
        self.inter = InterClassAnalyzer(data_dict, config)
    
    def load_data(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """
        Load data into both analyzers.
        
        Args:
            data_dict: Dictionary mapping split names to DataFrames
        """
        self.intra.load_data(data_dict)
        self.inter.load_data(data_dict)
    
    def complete_report(self, data_dict: Optional[Dict[str, pd.DataFrame]] = None) -> None:
        """
        Generate a complete report using both analyzers.
        
        Args:
            data_dict: Optional dictionary mapping split names to DataFrames
        """
        if data_dict is not None:
            self.load_data(data_dict)
        
        print("\nCOMPREHENSIVE WITHIN-CLASS ANALYSIS (INTRACLASS)")
        self.intra.report()
        
        print("\nCOMPREHENSIVE BETWEEN-CLASS ANALYSIS (INTERCLASS)")
        self.inter.report()
        