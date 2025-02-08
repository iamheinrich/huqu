from typing import List, Dict

class PromptTemplates:
    """Collection of prompt templates for different pipeline stages.
    
    The prompts follow this sequence:
    1. Caption Generation: Convert image to detailed description
    2. Dimension Discovery: Identify classification dimensions from captions
    3. Dimension Summary: Deduplicate and refine dimensions
    4. Feature Discovery: Define attributes for each dimension
    5. Attribute Summary: Deduplicate and refine attributes
    6. Classification: Assign captions to attributes
    7. Refinement: Handle edge cases and improve criteria
    8. Validation: Verify new attributes fit the criteria
    """
    
    # Stage 2: Caption Generation
    @staticmethod
    def image_caption(class_name: str = "the content") -> str:
        """Generate detailed caption for an image.
        
        Used in the Caption Generation stage to convert images into detailed textual descriptions.
        The caption should include the class name to ensure relevant details are captured.
        
        Args:
            class_name: Class or subject to focus on in the caption
        """
        return f"Please describe the image in detail with the keyword {class_name}"
    
    # Stage 3: Dimension Discovery
    @staticmethod
    def get_dimensions(main_subject: str, batch_size: int, caption_samples: str) -> str:
        """Discover potential dimensions for categorizing images.
        
        Used in the Dimension Discovery stage to identify meaningful ways to categorize the images.
        Processes captions in batches to suggest dimensions like color, pose, background, etc.
        
        Args:
            main_subject: Main subject/theme of the dataset
            batch_size: Number of captions in the current batch
            caption_samples: String of caption examples, newline separated
        """
        return (
            f"Chat, I want to categorize a set of images all about {main_subject} into several parts "
            f"and I'm asking you to suggest some dimensions I can use to categorize them. "
            f"For example, a dimension can be something like: color, position, background, or other keywords "
            f"that can better differentiate these images. Here are {batch_size} image captions: {caption_samples} "
            f"Please give me 5 most suggested dimensions with each represented by a single keyword in its lower case. "
            f"Answer strictly in the following format: 'Suggested Dimension: A, B, ...'"
        )
    
    @staticmethod
    def summarize_dimensions(dimensions: str) -> str:
        """Deduplicate and refine the discovered dimensions.
        
        Used after dimension discovery to remove redundancy and ensure dimensions are distinct.
        Combines similar dimensions and ensures they are at the right granularity level.
        
        Args:
            dimensions: Comma-separated string of dimensions to summarize
        """
        return (
            f"Here are several words each standing for a dimension to differentiate the images in a dataset, "
            f"but some of these words may be highly similar to each other and thus redundant. "
            f"Please give a refined set of classification dimensions strictly in the same format. "
            f"Here are the dimensions: {dimensions} "
            f"Answer strictly in the following format: 'Summarized Dimension: A, B, ...', "
            f"such as: 'Summarized Dimension: Action, Location, Mood, ...'"
        )
    
    # Stage 4: Feature Discovery
    @staticmethod
    def get_features(main_subject: str, dimension: str, caption_samples: str) -> str:
        """Discover attributes for a specific dimension.
        
        Used in the Feature Discovery stage to identify specific attributes for each dimension.
        Each attribute should be distinct and allow for clear classification.
        
        Args:
            main_subject: Main subject/theme of the dataset
            dimension: The dimension to find attributes for
            caption_samples: String of caption examples, newline separated
        """
        return (
            f"Chat, I want you to look at a few captions of images in my dataset about {main_subject} "
            f"and formulate a classification criteria for me to better structure my datasets with respect "
            f"to one classification dimension: {dimension}. For example, a good criteria I want you to craft "
            f"might be similar to 'action: blowing bubbles, fixing a bike, climbing...', where the word 'action' "
            f"stands for the dimension of this classification and the phrases 'blowing bubbles', 'fixing a bike', "
            f"'climbing'...stand for the specific action involved in the image described by the caption. "
            f"Here are the captions: {caption_samples} Please consider carefully about what to be included in "
            f"the criteria and only include the key phrases that can be frequently used. Each phrase should be "
            f"less than three words. Now give me the criteria strictly in the form of 'dimension: phrase1, phrase2, ...' "
            f"with at most 10 key phrases in total."
        )
    
    @staticmethod
    def summarize_attributes(dimension: str, suggestions: str) -> str:
        """Deduplicate and refine attributes for a dimension.
        
        Used after feature discovery to ensure attributes are distinct and non-overlapping.
        Each image should be classifiable into exactly one attribute per dimension.
        
        Args:
            dimension: The dimension these attributes belong to
            suggestions: String of attribute suggestions to summarize
        """
        return (
            f"Here are several attribute phrases with respect to one classification dimension: {dimension}, "
            f"but some of these phrases are highly similar to each other and thus redundant. "
            f"Please give a refined set of classification attribute phrases. "
            f"Here are the original attribute phrases: {suggestions} "
            f"Each phrase should be less than three words. "
            f"Answer strictly in the following format: 'Summarized Attributes: A, B, ...', "
            f"such as 'Summarized Attributes: blowing bubbles, fixing a bike, climbing...'"
        )
    
    # Stage 5: Classification
    @staticmethod
    def test_classification(dimension: str, sample: str, features: str) -> str:
        """Test classification of a caption into attributes. 
        
        Used in the Classification stage to assign each caption to exactly one attribute
        per dimension. May fail if no suitable attribute is found.
        
        Args:
            dimension: The dimension to classify for
            sample: The caption to classify
            features: Available features/attributes for this dimension
        """
        return (
            f"Chat, help me to classify the following caption into a classification criteria with respect to "
            f"the classification dimension of {dimension} please. The caption is: {sample} "
            f"The classification criteria is: {features} You have to pick up one among these choices. "
            f"Give me the answer strictly in the form of {{keyword}} if you think the caption belongs to "
            f"this keyword if categorized by this criteria. If you really don't think there is one answer, "
            f"reply and only reply '{{Unacceptable Criteria!}}'. Give me and only give me the answer."
        )
    
    # Stage 6: Refinement
    @staticmethod
    def refine_criteria(sample: str, criteria: str, test_results: str) -> str:
        """Refine classification criteria based on failed cases.
        
        Used in the Refinement stage to handle cases where classification fails.
        Can identify hallucinations, hard cases, redundant attributes, or missing attributes.
        
        Args:
            sample: The caption that failed classification
            criteria: Current classification criteria
            test_results: Results from testing the criteria
        """
        return (
            f"Hi, Chat! I'm trying to classify the following caption into a classification criteria. "
            f"However, it seems the current criteria fails to capture certain details that would help "
            f"differentiate this caption.\n\n"
            f"Caption: {sample}\n"
            f"Current Criteria: {criteria}\n"
            f"Test Results: {test_results}\n\n"
            f"We are unable to classify this caption using the provided criteria due to one of the "
            f"following reasons:\n\n"
            f"LLM Hallucination: If you believe the current criteria is reasonable, and the sample can be "
            f"classified under one of them, the current failure may be due to LLM hallucination. If the "
            f"majority of classifications are correct and only a small portion of the results appears highly "
            f"unreasonable, it is likely due to hallucination. In this case, please do nothing.\n\n"
            f"Answer format: {{\"hallucination\": []}}\n"
            f"Hard Case: If each classification result is reasonable on its own, but there are inconsistencies "
            f"between them (i.e., the results differ but none are unreasonable), this situation may be considered "
            f"a \"hard case\" where there is no clear-cut classification. In this case, please do nothing.\n\n"
            f"Answer format: {{\"hard_case\": []}}\n"
            f"Missing Attributes: If some important attributes are missing and need to be added to the criteria "
            f"to accurately classify the caption, please suggest one attribute to add.\n\n"
            f"Answer format: {{\"missing\": [\"keyword\"]}}"
        )
    
    @staticmethod
    def validate_attribute(criteria: str, new_attribute: str) -> str:
        """Validate if a new attribute fits the criteria.
        
        Used in the Refinement stage to verify that any new or merged attributes
        are consistent with the existing criteria.
        
        Args:
            criteria: Current classification criteria
            new_attribute: New attribute to validate
        """
        return (
            f"Chat, I'm refining a classification criteria with one extra attribute. "
            f"Tell me if this new attribute fits into the original criteria. "
            f"The current criteria is: {criteria} The new attribute is: {new_attribute} "
            f"Tell me in the strict form of {{Yes}} or {{No}}."
        )

    # Stage 7: Final Assignment
    @staticmethod
    def assign_attribute(dimension: str, caption: str, features: str) -> str:
        """Assign attribute to a caption during final classification.
        
        Used in the last pipeline stage after criteria stabilization.
        Simpler version without failure mode handling for bulk processing.
        
        Args:
            dimension: Finalized classification dimension
            caption: Caption text to classify
            features: Validated attribute options
        """
        return (
            f"Chat, help me to classify the following caption into a classification criteria "
            f"with respect to the classification dimension of {dimension} please. "
            f"The caption is: {caption} The classification criteria is: {features} "
            f"You have to pick up one among these choices. "
            f"Give me the answer strictly in the form of {{keyword}} "
            f"if you think the caption belongs to this keyword if categorized by this criteria. "
            f"Give me and only give me the answer:"
        )
