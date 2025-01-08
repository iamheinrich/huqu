from datasets import load_dataset, load_dataset_builder, get_dataset_infos, get_dataset_split_names
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch  # Added to handle tensor operations
import matplotlib.pyplot as plt

prompts = {
    "gender": {
        "single_gender": [
            "An image of a man",
            "An image of a woman"
        ],
        "multiple_one_gender": [
            "An image containing multiple men",
            "An image containing multiple women"
        ],
        "mixed_genders": [
            "An image containing both a man and a woman",
            "An image containing both men and women"
        ],
        "proportions": [
            "An image with more men than women",
            "An image with more women than men"
        ],
        "neutral_uncertain": [
            "An image where gender is not clearly identifiable"
        ]
    },
    "ethnicity": {
        "single_ethnicity": [
            "An image of a White person",
            "An image of a Black person",
            "An image of an Asian person",
            "An image of a Hispanic or Latino/a person",
            "An image of a Middle Eastern person",
            "An image of an Indian person"
        ],
        "multiple_ethnicities": [
            "An image containing people of different ethnicities",
            "An image containing multiple people, primarily White",
            "An image containing multiple people, primarily Black",
            "An image containing multiple people, primarily Asian",
            "An image containing multiple people, primarily Hispanic or Latino/a"
        ],
        "proportions": [
            "An image with more people of color than White people",
            "An image with more White people than people of color",
            "A diverse group of people from multiple ethnic backgrounds"
        ],
        "neutral_uncertain": [
            "An image where no specific ethnicity is clearly identifiable"
        ]
    },
    "age": {
        "finer_single_age_group": [
            "An image of a baby",
            "An image of a toddler",
            "An image of a child",
            "An image of a teenager",
            "An image of a young adult",
            "An image of a middle-aged adult",
            "An image of an elderly adult",
            "An image of a very old adult"
        ],
        "coarser_single_age_group": [
            "An image of a kid",
            "An image of a grown-up",
            "An image of an older adult"
        ],
        "multiple_age_groups": [
            "An image containing both children and adults",
            "An image containing a wide range of ages"
        ],
        "proportions": [
            "An image with mostly children",
            "An image with mostly older adults",
            "An image with mostly middle-aged adults"
        ],
        "neutral_uncertain": [
            "An image where age is not clearly identifiable"
        ]
    },
    "weight": {
        "single_body_type": [
            "An image of a thin person",
            "An image of an average-weight person",
            "An image of a plus-sized person",
            "An image of an underweight person"
        ],
        "multiple_body_types": [
            "An image containing people of various body sizes",
            "An image containing multiple thin people",
            "An image containing multiple plus-sized people"
        ],
        "proportions": [
            "An image with more plus-sized people than thin people",
            "An image with more thin people than plus-sized people"
        ],
        "neutral_uncertain": [
            "An image where body size is not clearly identifiable"
        ]
    }
}


def load_custom_dataset(dataset_name: str):
    """
    Loads the specified dataset using the datasets library.

    Args:
        dataset_name (str): The name of the dataset to load.

    Returns:
        Dataset: The loaded dataset.
    """
    print(f"Loading dataset {dataset_name}")
    dataset = load_dataset(dataset_name)
    print(f"Dataset loaded successfully")
    return dataset

def load_clip_model():
    """
    Loads the CLIP model and processor from the Hugging Face transformers library.

    Returns:
        tuple: A tuple containing the model and processor.
    """
    print("Loading CLIP model")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print("CLIP model loaded successfully")
    return model, processor

def get_dataset_information(dataset_name: str):
    """
    Retrieves information about the specified dataset.

    Args:
        dataset_name (str): The name of the dataset.

    Returns:
        dict: Information about the dataset.
    """
    print(f"Getting information about dataset {dataset_name}")
    ds_builder = load_dataset_builder(dataset_name)
    info_description = ds_builder.info.description
    feature_description = ds_builder.info.features
    dataset_infos = get_dataset_infos(dataset_name)
    split_names = get_dataset_split_names(dataset_name)
    return {
        "description": info_description,
        "features": feature_description,
        "infos": dataset_infos,
        "split_names": split_names
    }

def tag_images(dataset, model, processor, split="test"):
    """
    Tags images in the dataset using the CLIP model.

    Args:
        dataset (Dataset): The dataset containing images.
        model (CLIPModel): The CLIP model.
        processor (CLIPProcessor): The CLIP processor.
    """
    protected_attributes = ["gender", "ethnicity", "age", "weight"]
    
    # TODO: Let the user choose the protected attributes to tag the images.
    print("Tagging images")
    for image_data in dataset[split]: 
        image = image_data["image"]
        
        gender_prompts_single_gender = prompts["gender"]["single_gender"]
        gender_prompts_multiple_one_gender = prompts["gender"]["multiple_one_gender"]
        gender_prompts_mixed_genders = prompts["gender"]["mixed_genders"]
        gender_prompts_proportions = prompts["gender"]["proportions"]
        gender_prompts_neutral_uncertain = prompts["gender"]["neutral_uncertain"]
        
        gender_prompts = gender_prompts_single_gender + gender_prompts_multiple_one_gender + gender_prompts_mixed_genders + gender_prompts_proportions + gender_prompts_neutral_uncertain
        
        # Determine probabilities of the protected attributes
        inputs = processor(text=gender_prompts, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]
        print("Detached probs", probs)
        
        # Get the index of the highest probability
        max_prob = probs.max()
        max_idx = probs.argmax()
        predicted_tag = gender_prompts[max_idx]
        
        # Plot the image and predicted tag
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Display the image
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(f'Predicted tag: {predicted_tag} (Probability: {max_prob:.4f})', fontsize=24, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # Display the predicted tag
        print(f"Predicted tag: {predicted_tag} with probability {max_prob.item():.4f}")
    print("Images tagged successfully")
    
def main():
    # Load the dataset
    dataset = load_custom_dataset("rishitdagli/cppe-5")
    # TODO: Load the dataset automatically from data derived by website extraction or the information available from datasets library
    
    # Load the CLIP model
    model, processor = load_clip_model()
    
    # Get dataset information
    dataset_info = get_dataset_information("rishitdagli/cppe-5")
    # TODO: Determine if the dataset contains vulnerable attributes.
    
    # TODO: To start with, we can check if the dataset contains human. 
    # If so, we can create tags to classify the humans into protected groups such as gender, ethnicity, age, weight.
    
    # Tag the images
    tag_images(dataset, model, processor)
    
    # TODO: Evaluate the distribution of the tags

if __name__ == "__main__":
    main()
