from typing import Dict, Tuple, Optional, List, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

from huqu.analyzer.combined_analyzer import BaseAnalyzer


class IntraClassAnalyzer(BaseAnalyzer):
    """Analyzes attribute distributions within classes across data splits."""
    
    def report(self) -> None:
        """Generate and display comprehensive analysis report."""
        if self.data_dict is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        print("=" * 80)
        print("INTRACLASS ANALYSIS SUMMARY")
        print("=" * 80)

        stats = self._get_statistics()

        self._print_section("DATA SPLITS DISTRIBUTION", stats["split_level_distribution"])
        
        self._print_section("CLASS DISTRIBUTION", stats["class_level_distribution"])
        
        self._print_section("SUBPOPULATION SUMMARY", stats["subpopulation_summary"])

        self._print_section("ATTRIBUTE DISTRIBUTION", stats["attribute_level_distribution"])

        # Handle outliers
        outliers = self.detect_outliers()
        if outliers is not None:
            self._print_outlier_summary(outliers)
            display(outliers)
        else:
            print("\nOutlier analysis failed.")

        print("\nTip: Use get_class_outliers() or get_dimension_outliers() for detailed analysis and plot_histogram() to visualize its output.\n")
    
    def detect_outliers(self, over_threshold: Optional[float] = None, 
                        under_threshold: Optional[float] = None) -> Optional[pd.DataFrame]:
        """
        Detect attribute outliers across all splits.
        
        Args:
            over_threshold: Threshold for overrepresented attributes (default: from config)
            under_threshold: Threshold for underrepresented attributes (default: from config)
            
        Returns:
            DataFrame or None: Analysis results or None if analysis failed
        """
        if self.data_dict is None:
            raise ValueError("No data loaded. Call load_data() first.")

        over = over_threshold if over_threshold is not None else self.config.over_threshold
        under = under_threshold if under_threshold is not None else self.config.under_threshold
        
        try:
            return self._analyze_splits(over, under)
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            return None
        
    def get_class_outliers(self, class_name: str, over_threshold: Optional[float] = None, 
                         under_threshold: Optional[float] = None) -> Optional[pd.DataFrame]:
        """
        Get outlier analysis for a specific class.
        
        Args:
            class_name: Class to analyze
            over_threshold: Threshold for overrepresented attributes (default: from config)
            under_threshold: Threshold for underrepresented attributes (default: from config)
            
        Returns:
            DataFrame or None: Filtered analysis results
        """
        results = self.detect_outliers(over_threshold, under_threshold)
        if results is None:
            return None
            
        filtered = results[results["class"] == class_name]
        return filtered.reset_index(drop=True) if not filtered.empty else None

    def get_dimension_outliers(self, class_name: str, dimension: str, 
                             over_threshold: Optional[float] = None, 
                             under_threshold: Optional[float] = None) -> Optional[pd.DataFrame]:
        """
        Get outlier analysis for a specific class and dimension.
        
        Args:
            class_name: Class to analyze
            dimension: Dimension to analyze
            over_threshold: Threshold for overrepresented attributes (default: from config)
            under_threshold: Threshold for underrepresented attributes (default: from config)
            
        Returns:
            DataFrame or None: Filtered analysis results
        """
        results = self.detect_outliers(over_threshold, under_threshold)
        if results is None:
            return None
            
        mask = (results["class"] == class_name) & (results["dimension"] == dimension)
        filtered = results[mask]
                           
        if filtered.empty:
            print(f"No data found for class '{class_name}' and dimension '{dimension}'.")
            return None
            
        return filtered.reset_index(drop=True)

    def plot_histogram(self, results: pd.DataFrame, figsize: Optional[Tuple[int, int]] = None) -> None:
        """
        Plot attribute distribution across splits.
        
        Args:
            results: DataFrame from get_class_outliers() or get_dimension_outliers()
            figsize: Figure size in inches (width, height) (default: from config)
        """
        if not {"dimension", "class"}.issubset(results.columns):
            raise ValueError("Results must contain 'dimension' and 'class' columns")

        class_name = results['class'].iloc[0]
        unique_dimensions = results['dimension'].unique()
        dimension = unique_dimensions[0] if len(unique_dimensions) == 1 else None
        plot_figsize = figsize if figsize is not None else self.config.figure_size
        
        attributes = results["attribute"]
        x = np.arange(len(attributes))
        width = 0.4
        splits = [col.split("_")[0] for col in results.columns if col.endswith("_flag")]

        plt.figure(figsize=plot_figsize)
        
        # Plot bars for each split
        for i, split in enumerate(splits):
            plt.bar(x + (i * width) - (width / 2),
                    results[f"{split}_count"],
                    width=width,
                    label=split.capitalize())

        plt.xticks(x, attributes, rotation=45, ha="right")
        plt.xlabel("Attribute")
        plt.ylabel("Count")
        
        # Set appropriate title
        title = f"Attributes for Class '{class_name}'"
        if dimension:
            title = f"{title} in Dimension '{dimension}'"
        plt.title(title)
            
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _analyze_splits(self, over_threshold: float, under_threshold: float) -> pd.DataFrame:
        """Analyze attribute proportions across all splits."""
        split_dfs = []

        for split_name, df in self.data_dict.items():
            grouped = df.groupby(["class", "dimension", "attribute"])
            freq = grouped.size().reset_index(name="count")
            
            totals = freq.groupby(["class", "dimension"])["count"].transform("sum")
            freq = freq.assign(total=totals, proportion=freq["count"] / totals)

            flags = self._flag_proportions(freq["proportion"], over_threshold, under_threshold)
            props_formatted = freq["proportion"].map(lambda p: f"{p*100:.2f}%")
            
            split_df = freq[["class", "dimension", "attribute"]].copy()
            split_df[f"{split_name}_flag"] = flags
            split_df[f"{split_name}_proportion"] = props_formatted
            split_df[f"{split_name}_count"] = freq["count"]
            
            split_dfs.append(split_df)
        
        result = split_dfs[0]
        for df in split_dfs[1:]:
            result = pd.merge(result, df, on=["class", "dimension", "attribute"], how="outer")

        self._fill_missing_values(result)
        return result
    
    def _fill_missing_values(self, df: pd.DataFrame) -> None:
        """Fill missing values in the result DataFrame."""
        defaults = {
            "_flag": self.config.flags["missing"],
            "_proportion": "NaN",
            "_count": 0
        }
        
        for split in self.data_dict:
            for suffix, default in defaults.items():
                col = f"{split}{suffix}"
                if col in df.columns:
                    df.loc[:, col] = df[col].fillna(default)
    
    def _flag_proportions(self, proportions: pd.Series, over_threshold: float, 
                        under_threshold: float) -> np.ndarray:
        """Classify proportions based on thresholds."""
        return np.select(
            [proportions > over_threshold, proportions < under_threshold],
            [self.config.flags["over"], self.config.flags["under"]],
            ""
        )

    def _print_outlier_summary(self, outliers: pd.DataFrame) -> None:
        """Print summary of outlier analysis."""
        summary = {
            "under": {},
            "over": {},
            "missing": {}
        }

        # Count different types for each split
        for split in self.data_dict:
            flag_col = f"{split}_flag"
            
            if flag_col in outliers.columns:
                for category, flag_value in [
                    ("under", self.config.flags["under"]),
                    ("over", self.config.flags["over"]),
                    ("missing", self.config.flags["missing"])
                ]:
                    summary[category][split] = (outliers[flag_col] == flag_value).sum()

        print("\n" + "-" * 30 + "\n Outlier Analysis \n" + "-" * 30)
        print("Number of 🚨 underrepresented attributes:", 
              ", ".join([f"{s}: {c}" for s, c in summary["under"].items()]))
        print("Number of ⚠️ overrepresented attributes:", 
              ", ".join([f"{s}: {c}" for s, c in summary["over"].items()]))
        print("Number of ❗ missing attributes:", 
              ", ".join([f"{s}: {c}" for s, c in summary["missing"].items()]))

    def _get_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics at different levels."""
        stats = {}
        
        split_stats = {split: df["image_id"].nunique() for split, df in self.data_dict.items()}
        stats["split_level_distribution"] = pd.DataFrame.from_dict(
            split_stats, orient="index", columns=["Total Images"])
        
        group_levels = [
            ("class_level_distribution", ["class"]),
            ("attribute_level_distribution", ["class", "dimension", "attribute"])
        ]
        
        for stat_name, groupby_cols in group_levels:
            level_stats = {}
            for split, df in self.data_dict.items():
                level_stats[split] = df.groupby(groupby_cols)["image_id"].nunique()
            
            stats[stat_name] = pd.DataFrame(level_stats)
        
        stats["subpopulation_summary"] = self._get_subpopulation_stats()
        
        return stats
    
    def _get_subpopulation_stats(self) -> List[str]:
        """Generate statistics about dimensions and attributes per class."""
        class_data = {}
        
        for df in self.data_dict.values():
            for class_name, class_df in df.groupby("class"):
                if class_name not in class_data:
                    class_data[class_name] = {
                        "dimensions": set(), 
                        "attributes": set()
                    }
                
                class_data[class_name]["dimensions"].update(class_df["dimension"].unique())
                class_data[class_name]["attributes"].update(class_df["attribute"].unique())

        return [
            f"Class: {cls}\n"
            f"Total unique dimensions: {len(data['dimensions'])}\n"
            f"Total unique attributes: {len(data['attributes'])}\n"
            for cls, data in sorted(class_data.items())
        ]
