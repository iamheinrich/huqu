from typing import Dict, Optional, Union
from functools import reduce
import pandas as pd
from IPython.display import display, HTML

from huqu.combined_analyzer import BaseAnalyzer

class InterClassAnalyzer(BaseAnalyzer):
    """Analyze and compare attributes across different classes and dimensions."""
    
    def report(self, data_dict: Optional[Dict[str, pd.DataFrame]] = None, 
              rare_threshold: Optional[int] = None) -> None:
        """
        Generate a concise analysis report.
        
        Args:
            data_dict: Optional dictionary mapping split names to DataFrames
            rare_threshold: Threshold below which attributes are considered rare (default: from config)
        """
        if data_dict is not None:
            self.load_data(data_dict)
        elif self.data_dict is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        threshold = rare_threshold if rare_threshold is not None else self.config.rare_threshold
        
        print("=" * 80)
        print("INTERCLASS ANALYSIS SUMMARY")
        print("=" * 80)
        
        self._print_section("Missing Attributes Analysis", "Attributes present in some classes but missing in others:")
        for cls in self.unique_classes:
            print(f"\nClass: {cls}")
            unique_df = self.get_unique_to_class(cls)
            if isinstance(unique_df, pd.DataFrame):
                display(HTML(unique_df.to_html(index=False)))
            else:
                print(unique_df)
        
        self._print_section("Rare Attributes Analysis", f"Attributes appearing fewer than {threshold} times:")
        rare_attrs = self.analyze_rare_attributes(threshold)
        display(HTML(rare_attrs.to_html(index=False)))
    
    def analyze_rare_attributes(self, threshold: Optional[int] = None) -> pd.DataFrame:
        """
        Identify attributes appearing fewer times than the threshold.
        
        Args:
            threshold: Threshold value for rare attributes (default: from config)
            
        Returns:
            DataFrame: Rare attributes and their counts
        """
        if self.data_dict is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        th = threshold if threshold is not None else self.config.rare_threshold
            
        count_dfs = [
            df.groupby(['class', 'dimension', 'attribute']).size().reset_index(name=f'count_{name}')
            for name, df in self.data_dict.items()
        ]
        
        merged_df = reduce(
            lambda left, right: pd.merge(
                left, right, on=['class', 'dimension', 'attribute'], how='outer'
            ), 
            count_dfs
        )
        
        # Process and filter for rare attributes
        merged_df.fillna(0, inplace=True)
        count_cols = [col for col in merged_df.columns if col.startswith('count_')]
        merged_df[count_cols] = merged_df[count_cols].astype(int)
        
        merged_df['total_count'] = merged_df[count_cols].sum(axis=1)
        
        return merged_df[merged_df['total_count'] < th].sort_values(
            ['total_count', 'class', 'dimension']
        ).reset_index(drop=True)
    
    def compare_class_attributes(self) -> pd.DataFrame:
        """
        Compare attributes between classes and identify missing attributes.
        
        Returns:
            DataFrame: Missing attributes by class with statistics
        """
        if self.data_dict is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        combined_df = pd.concat(self.data_dict.values(), ignore_index=True)
        results = []

        for dimension in self.unique_dimensions:
            class_attrs = {
                cls: set(combined_df.loc[
                    (combined_df['class'] == cls) & 
                    (combined_df['dimension'] == dimension), 
                    'attribute'
                ])
                for cls in combined_df.loc[combined_df['dimension'] == dimension, 'class'].unique()
            }
            
            for cls, attrs in class_attrs.items():
                other_classes = {k: v for k, v in class_attrs.items() if k != cls}
                if not other_classes:
                    continue
                    
                all_other_attrs = set.union(*other_classes.values())
                missing_attrs = all_other_attrs - attrs
                
                for attr in missing_attrs:
                    present_in_classes = [k for k, v in other_classes.items() if attr in v]
                    
                    attr_filter = (
                        (combined_df['dimension'] == dimension) & 
                        (combined_df['attribute'] == attr)
                    )
                    
                    total_frequency = 0
                    total_samples = 0
                    
                    for c in present_in_classes:
                        class_filter = combined_df['class'] == c
                        frequency = len(combined_df[class_filter & attr_filter])
                        samples = len(combined_df[class_filter & (combined_df['dimension'] == dimension)])
                        
                        total_frequency += frequency
                        total_samples += samples
                    
                    avg_frequency = (total_frequency / total_samples * 100) if total_samples > 0 else 0
                    
                    results.append({
                        'dimension': dimension,
                        'missing in class': cls,
                        'missing attribute': attr,
                        'present in classes': ', '.join(present_in_classes),
                        'occurrence rate (%)': round(avg_frequency, 2)
                    })
        
        if not results:
            return pd.DataFrame()
            
        return pd.DataFrame(results).sort_values(
            ['dimension', 'occurrence rate (%)'], 
            ascending=[True, False]
        ).reset_index(drop=True)
    
    def get_unique_to_class(self, class_name: str) -> Union[pd.DataFrame, str]:
        """
        Get attributes missing in a specified class.
        
        Args:
            class_name: Name of the class to analyze
            
        Returns:
            DataFrame or str: Missing attributes or informational message
        """
        if self.data_dict is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        comparison_df = self.compare_class_attributes()
        
        if comparison_df.empty:
            return "No missing attributes found for any class."
            
        missing_attrs = comparison_df[comparison_df['missing in class'] == class_name]
        
        if missing_attrs.empty:
            return f"No missing attributes found for {class_name} in shared dimensions."
        
        return missing_attrs
