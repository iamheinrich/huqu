from datasets import load_dataset, load_dataset_builder, get_dataset_infos, get_dataset_split_names
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch  # Added to handle tensor operations
import matplotlib.pyplot as plt
from collections import Counter
from matplotlib import pyplot as plt
from cleanvision import Imagelab
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


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
    Returns:
        dict: A dictionary where keys are attributes and values are lists of predicted tags.
    """

    protected_attributes = ["gender", "ethnicity", "age", "weight"]

    while True:
        try:
            input_attributes = input(f"Enter one or more of the following attributes {protected_attributes} you would like to analyze (comma-separated): ").split(",")
            selected_attributes = [attribute.strip() for attribute in input_attributes]
    
            # check if input attributes are part of protected attributes
            valid_attributes = [attribute for attribute in selected_attributes if attribute in protected_attributes]
            if not valid_attributes:
                # currently raised only when all input attributes are not in protected attr
                raise ValueError("No valid attributes selected. Please try again.")
            
            print(f"Tagging images with selected attributes: {valid_attributes}")
            break

        except ValueError as e:
            print(e) 

    # Create dict where keys are attributes and values are lists of predicted tags 
    predicted_tags = {attribute: [] for attribute in valid_attributes}

    for image_data in dataset[split]: 
        image = image_data["image"]

        for attribute in valid_attributes:
            attribute_prompts = [prompt for category in prompts[attribute].values() for prompt in category]
            inputs = processor(text=attribute_prompts, images=image, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]

            max_idx = probs.argmax()
            predicted_tag = attribute_prompts[max_idx]
            
            # Add predictions to dict
            predicted_tags[attribute].append(predicted_tag)

    print("Images tagged successfully.")
    return predicted_tags

    # print("Probabilities for each prompt:")
    # for prompt, prob in zip(gender_prompts, probs):
    #     print(f"Prompt: '{prompt}' - Probability: {prob:.4f}")
    # print("Detached probs", probs)
    
    # Plot the image and predicted tag
    #fig, ax = plt.subplots(figsize=(8, 8))
    
    # Display the image
    #ax.imshow(image)
    #ax.axis('off')
    #ax.set_title(f'Predicted tag: {predicted_tag} (Probability: {max_prob:.4f})', fontsize=10, fontweight='bold')
    
    #plt.tight_layout()
    #plt.show()
    
    # Display the predicted tag
    # print(f"Predicted tag: {predicted_tag} with probability {max_prob.item():.4f}\n")
    # print(f"Images tagged successfully\n")
    # return predicted_tags
    
def evaluate_tag_distribution(predicted_tags: list):
    """
    Evaluates and visualizes the distribution of predicted tags for each attribute.
    
    Args:
        predicted_tags (dict): A dictionary where keys are attributes and values are lists of predicted tags.
    Returns:
        None
    """
    
    for attribute, tags in predicted_tags.items():
        print(f"\nAnalyzing distribution for attribute: {attribute.capitalize()}")
        tag_counts = Counter(tags)
        total_tags = sum(tag_counts.values())
        tags_percentages = {tag: (count / total_tags) * 100 for tag, count in tag_counts.items()}

        # Plot tag distribution as a bar chart
        plt.figure(figsize=(16, 9))
        plt.bar(tags_percentages.keys(), tags_percentages.values(), color="c")
        plt.xlabel("Assigned tags")
        plt.ylabel("Frequency (%)")
        plt.title(f"Distribution of Predicted Tags for {attribute.capitalize()}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

        print("Tag Distribution (Percentages):")
        for tag, proportion in tags_percentages.items():
            print(f"{tag}: {proportion:.2f}%")

def summarize_image_issues(available_splits: list):

    issue_summary = pd.DataFrame()

    for split in available_splits:
        dataset = load_dataset("rishitdagli/cppe-5", split=split)
        imagelab = Imagelab(hf_dataset=dataset, image_key="image")
        imagelab.find_issues()
        
        issue_counts = imagelab.issues.filter(like="is_").sum()
        issue_percentage = round((issue_counts / len(imagelab.issues)) * 100, 3)
        
        issue_summary[split] = issue_counts
        issue_summary[f"{split}_percentage"] = issue_percentage
    
    print(issue_summary)

def main():
    # Load the dataset
    dataset = load_custom_dataset("rishitdagli/cppe-5")
    # TODO: Load the dataset automatically from data derived by website extraction or the information available from datasets library
    # Load the CLIP model
    model, processor = load_clip_model()
    
    # Get dataset information
    dataset_info = get_dataset_information("rishitdagli/cppe-5")
    # print(dataset_info)
    
    # TODO: Determine if the dataset contains vulnerable attributes.
    
    # TODO: Detect if dataset contains people
    # If so, we can create tags to classify the humans into protected groups such as gender, ethnicity, age, weight.
    
    # Tag the images
    predicted_tags = tag_images(dataset, model, processor)

    evaluate_tag_distribution(predicted_tags)

    summarize_image_issues(dataset_info["split_names"])

if __name__ == "__main__":
    main()
