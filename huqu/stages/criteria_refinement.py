from typing import Any, Dict, List, Optional
import pandas as pd
from dataclasses import dataclass
import logging
from tqdm import tqdm
from .base import PipelineStage
from ..models.base import BaseModel
from ..prompts.templates import PromptTemplates
import json

logger = logging.getLogger(__name__)
@dataclass
class RefinementResult:
    """Result of a single refinement iteration."""
    dimension: str
    caption: str
    failure_type: str
    suggested_changes: List[str]
    was_applied: bool = False


class CriteriaRefinementStage(PipelineStage):
    """Pipeline stage for refining classification criteria through iterative testing.
    
    Overview of the simplified approach:
      1. Load unrefined criteria from configured path
      2. Randomly sample dataset captions
      3. Test each caption against all dimensions
      4. Attempt refinements if classification fails
      5. Validate and apply refinements
      6. Repeat until convergence or max rounds reached
      7. Save refined criteria to configured path
    
    The stage loads unrefined criteria from unrefined_criteria_path and saves
    the refined version to refined_criteria_path as JSON.
    """
    
    def __init__(self, model: BaseModel):
        """Initialize the criteria refinement stage.
        
        Args:
            model: Language model for classification and refinement
        """
        super().__init__(models={"llm": model})
        self.num_refining_rounds = self.config["stages"]["criteria_refinement"]["num_rounds"]
        self.sample_size = self.config["stages"]["criteria_refinement"]["sample_size"]

    def process(self, **kwargs) -> Dict[str, Any]:
        """Perform iterative refinement of the classification criteria.
        
        Loads unrefined criteria from the configured unrefined_criteria_path,
        refines them through iterative testing against caption samples, and
        saves the refined criteria to the configured refined_criteria_path as JSON.
        
        Returns:
            Dict containing the refined criteria with structure:
            {
              "dimensions": [...],
              "attributes": {
                  dim: [...],
                  ...
              }
            }
            Note: While this method returns the criteria, it also saves them to the
            configured path for use by subsequent stages.
        """
        # Load the dataset and unrefined criteria
        df = pd.read_parquet(self.config["dataset"]["captions_path"])
        with open(self.config["dataset"]["unrefined_criteria_path"]) as f:
            criteria = json.load(f)

        # Extract dimension -> attributes from existing criteria
        dimensions = {
            dim: criteria["attributes"][dim]
            for dim in criteria["dimensions"]
        }
        
        # Perform iterative refinement
        pbar = tqdm(range(self.num_refining_rounds), desc="Refinement rounds")
        for round_idx in pbar:
            round_captions = self._sample_captions(df)
            refinements = []
            
            # Test classification for each sample and gather refinements
            for caption in tqdm(round_captions, desc=f"Processing captions (round {round_idx + 1})", leave=False):
                caption_refinements = self._process_caption(caption, dimensions)
                refinements.extend(caption_refinements)
            
            # Apply any refinements
            dimensions = self._apply_refinements(dimensions, refinements)
            
            # Update progress bar with refinement info
            num_applied = sum(1 for r in refinements if r.was_applied)
            pbar.set_postfix({"applied_refinements": num_applied})
            
            # If no refinements were applied, we assume convergence and stop
            if not any(r.was_applied for r in refinements):
                logger.info(f"Converged after {round_idx + 1} rounds")
                break

        # Build final output
        refined_criteria = {
            "dimensions": list(dimensions.keys()),
            "attributes": dimensions
        }

        # Save refined criteria
        output_path = self.config["dataset"]["refined_criteria_path"]
        with open(output_path, "w") as f:
            json.dump(refined_criteria, f, indent=2)

        return refined_criteria

    def _sample_captions(self, df: pd.DataFrame) -> List[str]:
        """
        Randomly sample captions from the dataset for refinement.
        """
        return df["caption"].sample(
            n=min(self.sample_size, len(df)),
            random_state=42  # For reproducibility
        ).tolist()

    def _process_caption(
        self, 
        caption: str, 
        dimensions: Dict[str, List[str]]
    ) -> List[RefinementResult]:
        """
        Test how well the current dimensions classify a single caption. 
        If classification fails, propose refinements.
        """
        results = []
        for dimension, features in dimensions.items():
            # Test classification
            test_result = self._test_classification(caption, dimension, features)
            
            # If classification failed, try to refine
            if test_result == "Unacceptable Criteria!":
                # Join dimension and features as criteria
                refinement = self._refine_criteria(caption, dimension, features)
                if refinement:
                    results.append(refinement)
        return results

    def _test_classification(
        self, 
        caption: str, 
        dimension: str, 
        features: List[str]
    ) -> str:
        """
        Test if a caption can be classified under current criteria.
        Returns either the chosen feature or "Unacceptable Criteria!".
        """
        features_str = ", ".join(features)
        prompt = PromptTemplates.test_classification(
            dimension=dimension,
            sample=caption,
            features=features_str
        )
        result = self.models["llm"].generate(prompt)
        
        # The prompt expects the model to return {some_feature}
        # or {Unacceptable Criteria!}, so remove braces:
        return result.strip("{}")

    def _refine_criteria(
        self,
        caption: str,
        dimension: str,
        features: List[str]
    ) -> Optional[RefinementResult]:
        """
        Ask the LLM to refine classification criteria if a caption fails to classify.
        Possible responses:
          - HALLUCINATION (do nothing)
          - HARD_CASE (do nothing)
          - MISSING: new_attribute (attempt to add)
        """
        criteria_str = f"{dimension}: {', '.join(features)}"
        prompt = PromptTemplates.refine_criteria(
            sample=caption,
            criteria=criteria_str,
            test_results="Unacceptable Criteria!"
        )
        
        refinement_response = self.models["llm"].generate(prompt)
        clean_response = refinement_response.strip()
        upper_response = clean_response.upper()

        # 1) MISSING: ...
        if upper_response.startswith("MISSING:"):
            parts = clean_response.split(":", 1)
            if len(parts) < 2:
                logger.warning(f"MISSING: but no attribute found. Response: {clean_response}")
                return None

            new_feature = parts[1].strip().split("\n")[0].strip()
            new_feature = new_feature.strip('"\'{}').strip()
            if not new_feature:
                logger.warning(f"No actual feature after 'MISSING:'. Response: {clean_response}")
                return None

            return RefinementResult(
                dimension=dimension,
                caption=caption,
                failure_type="missing",
                suggested_changes=[new_feature]
            )

        # 2) HALLUCINATION or HARD_CASE
        if upper_response in {"HALLUCINATION", "HARD_CASE"}:
            return None

        # 3) Anything else is unrecognized
        logger.warning(f"Unrecognized refinement response format: {clean_response}")
        return None

    def _apply_refinements(
        self, 
        dimensions: Dict[str, List[str]], 
        refinements: List[RefinementResult]
    ) -> Dict[str, List[str]]:
        """
        Apply any validated refinements to the dimension-attribute map.
        Right now, we only handle missing attributes.
        """
        for refinement in refinements:
            if refinement.failure_type == "missing":
                dimension = refinement.dimension
                current_features = dimensions[dimension]

                for new_feature in refinement.suggested_changes:
                    # Check for duplicates (case-insensitive)
                    if new_feature.lower() not in (f.lower() for f in current_features):
                        # Validate
                        if self._validate_attribute(current_features, new_feature):
                            current_features.append(new_feature)
                            refinement.was_applied = True
        
        return dimensions

    def _validate_attribute(
        self, 
        current_features: List[str], 
        new_feature: str
    ) -> bool:
        """
        Check if adding this new_feature makes sense under the current feature set.
        The LLM should respond {yes} or {no} if consistent.
        """
        features_str = ", ".join(current_features)
        prompt = PromptTemplates.validate_attribute(
            criteria=features_str,
            new_attribute=new_feature
        )
        result = self.models["llm"].generate(prompt)
        return result.strip("{}").strip().lower() == "yes"
