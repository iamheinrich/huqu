from datasets import load_dataset, get_dataset_infos
from huqu.pipeline import SubpopulationPipeline, PipelineConfig

def main():
    # Load just 2 images from CIFAR-10 dataset
    dataset_name = "cifar10"
    dataset = load_dataset(dataset_name, split="train[:300]")  # Use only 2 images for quick testing
    dataset.name = dataset_name  # Set the dataset name for image key detection
    
    print(f"\nProcessing {len(dataset)} images...")
    
    # Create pipeline with default config
    pipeline = SubpopulationPipeline()
    
    # Process the dataset
    results = pipeline.process_dataset(dataset)
    
    # Print results summary
    print("\nResults:")
    print("Captions generated:", results["captions"])
    print("\nDiscovered dimensions:", list(results["dimensions"].keys()))
    print("\nDimension categories:", results["dimensions"])
    print("\nAssigned subpopulations:", results["subpopulations"])
    
    # Generate visualizations
    summary = pipeline.visualize_subpopulations(results)
    print("\nSubpopulation Summary:")
    for dim, counts in summary.items():
        print(f"\n{dim}:")
        print(counts)

if __name__ == "__main__":
    main() 