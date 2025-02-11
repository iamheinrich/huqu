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
    def summarize_and_reduce_dimensions(dimensions: List[str]) -> str:
        """
        Merge or remove redundant/overly similar dimensions. 
        Keep the most general or broadly applicable dimension wherever possible. 
        Return final result in the format:
        
            Final Dimensions: dim1, dim2, dim3, ...
        """

        dims_str = ", ".join(dimensions)

        return (
            f"Below is a list of discovered dimension keywords for classifying images, but they could be:\n"
            f"- Duplicative (e.g., 'color', 'hue', 'shade' all refer to color)\n"
            f"- Overly specific if there's a more general dimension available\n"
            f"- Irrelevant or hateful/harmful\n\n"

            # --- FEW-SHOT EXAMPLE 1 ---
            f"Example 1:\n"
            f"Given Dimensions: color, hue, shade, brightness\n"
            f"Analysis: 'hue' and 'shade' are essentially sub-concepts of 'color'; 'brightness' might be too narrow.\n"
            f"Decision: Merge everything into 'color' because it's the most general dimension.\n"
            f"Correct Response: Final Dimensions: color\n\n"

            # --- FEW-SHOT EXAMPLE 2 ---
            f"Example 2:\n"
            f"Given Dimensions: mood, emotion, atmosphere, color, color\n"
            f"Analysis: 'mood', 'emotion', and 'atmosphere' overlap heavily; 'color' is distinct. "
            f"There are duplicates of 'color'.\n"
            f"Decision: Merge mood, emotion, atmosphere into 'mood' (the broadest). Keep 'color' once.\n"
            f"Correct Response: Final Dimensions: mood, color\n\n"

            # --- FEW-SHOT EXAMPLE 3 ---
            f"Example 3:\n"
            f"Given Dimensions: background, location, setting, hateful_word\n"
            f"Analysis: 'hateful_word' is inappropriate; remove it. 'background' and 'setting' overlap with 'location'.\n"
            f"Decision: Merge 'background' and 'setting' into 'location' since it's broader.\n"
            f"Correct Response: Final Dimensions: location\n\n"

            # --- FEW-SHOT EXAMPLE 4 ---
            f"Example 4:\n"
            f"Given Dimensions: action, pose, posture\n"
            f"Analysis: 'pose' and 'posture' overlap; 'action' is somewhat distinct because it implies motion.\n"
            f"Decision: Merge 'pose' and 'posture' into 'pose'. Keep 'action' if it’s significantly different.\n"
            f"Correct Response: Final Dimensions: pose, action\n\n"

            # -- ACTUAL TASK --
            f"Now, here are the dimensions you need to refine: {dims_str}\n\n"
            f"Instructions:\n"
            f"1) Merge any dimensions that are synonyms or near-duplicates into a single keyword.\n"
            f"2) Remove any irrelevant or hateful/harmful dimension.\n"
            f"3) If multiple dimensions overlap, keep the one that is the most general or widely applicable.\n"
            f"4) Keep the final list under ~5 to 7 distinct dimensions.\n\n"
            f"Return your final answer in this strict format:\n"
            f"Final Dimensions: dimA, dimB, dimC, ..."
        )

    # Stage 4: Feature Discovery
    @staticmethod
    def get_features(main_subject: str, dimension: str, caption_samples: str) -> str:
        """
        Discover attributes for a specific dimension, in a dataset-agnostic way. Only include
        attributes directly relevant to the given dimension (e.g., if the dimension is "color,"
        you might include "red," "blue," "green," but ignore time references; if the dimension
        is "time_of_day," you might include "morning," "night," "sunset," but ignore colors or
        location references, etc.).

        Each attribute must:
        - Be less than three words
        - Be representative of the dimension in question
        - Avoid overlapping concepts from other potential dimensions
        - Occur frequently enough across the captions to be useful

        The final answer must be strictly in the format:
        dimension: phrase1, phrase2, phrase3, ...

        Args:
            main_subject: The overall subject/theme of the dataset (e.g., "scenes" or "portraits")
            dimension: The dimension to find attributes for (e.g., "color," "time_of_day," "location")
            caption_samples: A string of caption examples (newline-separated)
        """

        return (
            f"I want you to analyze these image captions about {main_subject}. We need to extract "
            f"a set of up to 10 key phrases that represent the single dimension '{dimension}'.\n\n"

            f"Here are the captions:\n"
            f"{caption_samples}\n\n"

            f"Guidelines:\n"
            f"- Focus exclusively on '{dimension}'. Do not include phrases that belong to other dimensions.\n"
            f"- Each phrase must be less than three words.\n"
            f"- For instance, if the dimension is 'color,' only list color attributes (e.g., 'red,' 'pale blue'); "
            f"if the dimension is 'time_of_day,' only list times or lighting-related phrases (e.g., 'morning,' 'sunset').\n"
            f"- Exclude any attributes that do not reflect the dimension '{dimension}'.\n\n"

            f"Finally, provide your answer in exactly this form (with no extra text):\n"
            f"dimension: phrase1, phrase2, phrase3, ...\n"
        )
    
    @staticmethod
    def summarize_and_reduce_attributes(dimension: str, attributes: List[str]) -> str:
        """
        Merge or remove redundant/overly similar attributes for a specific dimension.
        Keep at most ~5 that are broad or most commonly applicable. Remove hateful/harmful.
        Return final result in the format:
            Final Attributes: attr1, attr2, attr3, ...
        """

        attrs_str = ", ".join(attributes)

        return (
            f"Here is a list of attribute phrases for the dimension '{dimension}'. Some may be duplicates, "
            f"synonyms, overly specific, or harmful.\n\n"

            # --- FEW-SHOT EXAMPLE 1 ---
            f"Example 1:\n"
            f"Dimension: color\n"
            f"Attributes: white, black, turquoise, teal, red, pink, vibrant green, bright orange\n"
            f"Analysis: 'turquoise' and 'teal' overlap; 'vibrant green' can be shortened to 'green'. "
            f"Keep only 5 total.\n"
            f"Correct Response: Final Attributes: white, black, green, orange, red\n\n"

            # --- FEW-SHOT EXAMPLE 2 ---
            f"Example 2:\n"
            f"Dimension: activity\n"
            f"Attributes: sleeping, snoozing, lying down, resting, dozing, relaxing\n"
            f"Analysis: All revolve around 'sleeping'/'resting'; condense into fewer distinct items.\n"
            f"Correct Response: Final Attributes: sleeping, resting\n\n"

            # --- FEW-SHOT EXAMPLE 3 ---
            f"Example 3:\n"
            f"Dimension: shape\n"
            f"Attributes: round, circular, ball-shaped, square, quadrilateral, polygon\n"
            f"Analysis: 'round', 'circular', and 'ball-shaped' are nearly the same. 'square' and 'quadrilateral' "
            f"are quite similar. Keep only ~5 attributes that are distinct.\n"
            f"Correct Response: Final Attributes: round, square, polygon\n\n"

            # --- FEW-SHOT EXAMPLE 4 ---
            f"Example 4:\n"
            f"Dimension: texture\n"
            f"Attributes: rough, bumpy, smooth, smooth surface, grainy, coarse, silky, harsh\n"
            f"Analysis: 'smooth' and 'smooth surface' overlap. 'rough', 'bumpy', 'grainy', 'coarse', and 'harsh' "
            f"might be partially redundant; 'harsh' could be merged into 'rough' if appropriate. "
            f"Keep just a handful of distinct terms.\n"
            f"Correct Response: Final Attributes: rough, smooth, grainy, silky\n\n"

            f"Now, for the dimension '{dimension}', the attributes are: {attrs_str}\n\n"
            f"Guidelines:\n"
            f"1) Merge near-duplicates or synonyms.\n"
            f"2) Remove harmful or irrelevant items.\n"
            f"3) Keep at most ~5 broad or commonly used attributes.\n\n"
            f"Return them strictly in the format:\n"
            f"Final Attributes: A, B, C, ..."
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
            f"1. LLM Hallucination: If you believe the current criteria is reasonable, and the sample can be "
            f"classified under one of them, the current failure may be due to LLM hallucination. This occurs when "
            f"the majority of classifications are correct and only a small portion of results appear highly "
            f"unreasonable.\n\n"
            f"2. Hard Case: If each classification result is reasonable on its own, but there are inconsistencies "
            f"between them (i.e., the results differ but none are unreasonable), this situation may be considered "
            f"a \"hard case\" where there is no clear-cut classification.\n\n"
            f"3. Missing Attributes: If some important attributes are missing and need to be added to the criteria "
            f"to accurately classify the caption.\n\n"
            f"Respond with ONLY ONE of these formats:\n"
            f"- 'HALLUCINATION' (if reason 1 applies)\n"
            f"- 'HARD_CASE' (if reason 2 applies)\n"
            f"- 'MISSING: attribute_phrase' (if reason 3 applies)\n"
            f"Example: 'MISSING: nighttime lighting'"
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
            f"Here is a set of classification criteria: {criteria} "
            f"I want to add a new attribute: {new_attribute} "
            f"Please tell me if this new attribute is consistent with the existing criteria. "
            f"Answer strictly in the form of {{yes}} or {{no}}."
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