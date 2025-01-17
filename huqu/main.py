from datasets import load_dataset, load_dataset_builder, get_dataset_infos, get_dataset_split_names
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import torch  # Added to handle tensor operations
import matplotlib.pyplot as plt
from collections import Counter
from cleanvision import Imagelab
import pandas as pd
from tqdm import tqdm

# TODO: reorganize prompt structure
prompts = {
    "gender": {
        # "single_gender": [
        #     "An image of a man",
        #     "An image of a woman"
        # ],
        # "multiple_one_gender": [
        #     "An image containing multiple men",
        #     "An image containing multiple women"
        # ],
        "male": [
            "An image of a man",
            "An image containing multiple men"
        ],
        "female": [
            "An image of a woman",
            "An image containing multiple women"
        ],
        "mixed_genders": [
            "An image containing both a man and a woman",
            "An image containing both men and women"
        ],
        # "proportions": [
        #     "An image with more men than women",
        #     "An image with more women than men"
        # ],
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

def choose_attributes():

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
            
            break

        except ValueError as e:
            print(e) 

    return valid_attributes

def tag_images(dataset, model, processor, selected_attributes, split):
    
    print(f"Tagging images with selected attributes: {selected_attributes}")
    # Create dict where keys are attributes and values are lists of predicted tags 
    predicted_tags = {attribute: [] for attribute in selected_attributes}

    for image_data in tqdm(dataset[split], desc="Tagging images", unit="image"): 
        image = image_data["image"]
    
        for attribute in selected_attributes:
            attribute_prompts = []
            prompt_to_group = {}
    
            # TODO: reorganize prompt structure
            for group_name, prompts_list in prompts[attribute].items():
                for prompt in prompts_list:
                    attribute_prompts.append(prompt)
                    prompt_to_group[prompt] = group_name
    
            inputs = processor(text=attribute_prompts, images=image, return_tensors="pt", padding=True)
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).detach().numpy()[0]
    
            max_idx = probs.argmax()
            predicted_prompt = attribute_prompts[max_idx]
            predicted_group = prompt_to_group[predicted_prompt]
            
            # Add the predicted group to the resulting tags
            predicted_tags[attribute].append(predicted_group)


    print("Images tagged successfully.")
    return predicted_tags

def calculate_tag_counts_and_percentages(attribute, tags):
    """Calculate tag counts and percentages for a given attribute in a split."""
    if attribute not in tags:
        return {}, {}

    tag_counts = Counter(tags[attribute])
    total_tags = sum(tag_counts.values())
    tag_percentages = {tag: round((count / total_tags) * 100, 2) for tag, count in tag_counts.items()}
    return tag_counts, tag_percentages

def analyze_attribute_data_per_split(attribute, predicted_tags):
    """Analyze tag distribution for an attribute across splits."""
    results = {}
    
    for split, tags in predicted_tags.items():
        tag_counts, tag_percentages = calculate_tag_counts_and_percentages(attribute, tags)
        update_results_with_split_data(results, split, tag_counts, tag_percentages)
    
    df = pd.DataFrame.from_dict(results, orient="index").fillna(0)
    df.index.name = "Tag"
    df.reset_index(inplace=True)
    df.sort_values(by="Tag", inplace=True)
    return df

def update_results_with_split_data(results, split, tag_counts, tag_percentages):
    """Update results dictionary with count and percentage data for given split."""
    for tag, count in tag_counts.items():
        if tag not in results:
            results[tag] = {}
        results[tag][f"Count_{split}"] = count
        results[tag][f"Percentage_{split}"] = tag_percentages[tag]

def generate_tagging_report(predicted_tags):
    """Generate tables for tag distributions across attributes."""
    
    all_attributes_tables = {}
    
    # Get attribute name to be analyzed
    attributes = next(iter(predicted_tags.values())).keys()
    
    for attribute in attributes:
        df = analyze_attribute_data_per_split(attribute, predicted_tags)
        
        # Print table for the current attribute
        print(f"\nTag Distribution for Attribute: {attribute.capitalize()}")
        print("=" * 90)
        print(df.to_string(index=False))
        print("=" * 90)
        
        # Store DataFrame in the results dictionary
        all_attributes_tables[attribute] = df
    
    return all_attributes_tables

def visualize_tag_percentages(attribute_tables):
    """Plot tag percentages as bar chart"""
    for attribute, df in attribute_tables.items():
        # Select percentage column for bar chart
        percent_cols = [col for col in df.columns if col.startswith("Percentage_")]
        percent_df = df.set_index("Tag")[percent_cols]
    
        # Uncomment to get top 4 tags across splits
        # percent_df["Total_Percentage"] = percent_df.sum(axis=1)
        # top_tags_distribution = percent_df.nlargest(4, "Total_Percentage").drop(columns=["Total_Percentage"])
        
        # Plot bar chart comparing splits
        percent_df.plot(kind="bar", stacked=False, figsize=(10, 6))
        plt.title(f"Percentage Distribution for attribute: {attribute.capitalize()}")
        plt.ylabel("Percentage (%)")
        plt.xlabel("Tag")
        plt.xticks(rotation=0)
        plt.legend(title="Split")
        plt.tight_layout()
        plt.show()

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
    
    print("\nImage Issues detected across splits:")
    print("=" * 90)
    print(issue_summary)
    print("=" * 90)

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
    
    chosen_attributes = choose_attributes()

    # Tag all images across splits with chosen_attributes
    all_predicted_tags = {}
    for split in dataset_info["split_names"]:
        print(f"\nProcessing split: {split}")
        # Tag images for current split using chosen_attributes
        predicted_tags = tag_images(dataset, model, processor, chosen_attributes, split=split)
        all_predicted_tags[split] = predicted_tags


    tagging_report = generate_tagging_report(all_predicted_tags)

    visualize_tag_percentages(tagging_report)

    summarize_image_issues(dataset_info["split_names"])

if __name__ == "__main__":
    main()
