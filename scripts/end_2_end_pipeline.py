#!/usr/bin/env python3
"""
End-to-end pipeline for generating Semantic Structure Descriptions (SSD).

This script demonstrates how to use the huqu package to:
1. Load and filter images for a specific class
2. Generate detailed captions for each image
3. Discover classification dimensions and attributes
4. Refine and validate the classification criteria
5. Assign images to attributes within each dimension
"""

import logging
from pathlib import Path
from typing import Optional

from huqu.stages.dataset_loading import DatasetLoadingStage
from huqu.stages.caption_generation import CaptionGenerationStage
from huqu.stages.criteria_initilization import CriteriaInitializationStage
from huqu.stages.criteria_refinement import CriteriaRefinementStage
from huqu.stages.image_assignment import ImageAssignmentStage
from huqu.models.chatgpt import GPT4oMiniMLLM, GPT4oMiniLLM

# Configure minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
# Suppress httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def run_pipeline(
    **kwargs
) -> None:
    """
    Run the complete SSD generation pipeline.

    Args:
        **kwargs: Additional configuration parameters for pipeline stages
    """

    try:
        # Initialize models
        logger.info("Starting pipeline...")
        multimodal_model = GPT4oMiniMLLM()  # For tasks requiring image understanding
        text_model = GPT4oMiniLLM()         # For text-only tasks

        # Dataset Loading
        logger.info("Loading dataset...")
        dataset_loader = DatasetLoadingStage()
        dataset = dataset_loader.process(**kwargs)

        # Caption Generation
        logger.info("Generating captions...")
        caption_generator = CaptionGenerationStage(multimodal_model)
        caption_generator.process(dataset)

        # Criteria Initialization
        logger.info("Initializing criteria...")
        criteria_initializer = CriteriaInitializationStage(text_model)
        criteria = criteria_initializer.process()

        # Criteria Refinement
        logger.info("Refining criteria...")
        criteria_refiner = CriteriaRefinementStage(text_model)
        criteria = criteria_refiner.process()

        # Image Assignment
        logger.info("Assigning images...")
        image_assignment = ImageAssignmentStage(text_model)
        image_assignment.process()

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    run_pipeline() 