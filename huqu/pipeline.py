from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datasets import Dataset, DatasetDict, get_dataset_infos
import torch
from transformers import pipeline, AutoProcessor
import pandas as pd
import numpy as np
from tqdm import tqdm
import signal
from contextlib import contextmanager
import time

# NEW: we import huggingface_hub + llama-cpp-python
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)

@dataclass
class PipelineConfig:
    """Configuration for the subpopulation discovery pipeline."""
    batch_size: int = 1  # Reduced to 1 for CPU
    max_dimensions: int = 5
    model_id: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    
    # CHANGED: The text model ID now points to the GGUF repo on Hugging Face
    text_model_id: str = "Qwen/Qwen2.5-3B-Instruct-GGUF"
    
    device: str = "cpu"  # Use CPU by default
    max_new_tokens: int = 20
    generation_timeout: int = 300
    torch_dtype: torch.dtype = torch.float32  # For the LLaVA pipeline

class SubpopulationPipeline:
    """Pipeline for discovering and analyzing subpopulations in image datasets."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._setup_models()
    
    def _setup_models(self):
        """Initialize LLaVA pipeline and processor, plus a llama-cpp-based Qwen model."""
        print(f"Initializing LLaVA-OneVision pipeline on {self.config.device}...")
        
        # Suppress unnecessary warnings
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # --------------------------------------------------
        # 1) LLaVA model for image-to-text
        # --------------------------------------------------
        self.image_text_model = pipeline(
            "image-text-to-text",
            model=self.config.model_id,
            device=self.config.device,
            torch_dtype=self.config.torch_dtype,
        )
        
        self.processor = AutoProcessor.from_pretrained(self.config.model_id)

        # --------------------------------------------------
        # 2) Qwen in GGUF format, via llama-cpp
        # --------------------------------------------------
        print("Initializing Qwen GGUF model for text-only queries...")

        # Download a specific .gguf file from the huggingface repo programmatically
        gguf_filename = "qwen2.5-3b-instruct-q4_0.gguf"
        
        model_path = hf_hub_download(
            repo_id=self.config.text_model_id,
            filename=gguf_filename,
            cache_dir=".cache"  # Cache locally to avoid re-downloading
        )
        
        # Create the llama-cpp model object with minimal logging
        import os
        os.environ["LLAMA_CPP_VERBOSE"] = "0"
        
        self.qwen_gguf = Llama(
            model_path=model_path,
            n_threads=4,
            n_ctx=512,
            verbose=False
        )

    def _extract_text_from_response(self, result: List[Dict]) -> str:
        """Extract the assistant's response from a conversation for the LLaVA output."""
        try:
            print("\nDebug - Raw model output:", result)
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', [])
                for message in generated_text:
                    if message.get('role') == 'assistant':
                        return message.get('content', "")
            return ""
        except Exception as e:
            print(f"Error extracting text from response: {str(e)}")
            return ""

    def _generate_caption(self, image) -> str:
        """Generate caption for a single image using LLaVA."""
        try:
            with timeout(self.config.generation_timeout):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the image in detail."},
                            {"type": "image", "image": image}
                        ]
                    }
                ]
                
                result = self.image_text_model(
                    text=messages,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False
                )
                
                if isinstance(result, list):
                    return self._extract_text_from_response(result)
                return "Error: Invalid model output"
                
        except TimeoutException:
            print("Caption generation timed out")
            return "Error: Caption generation timed out"
        except Exception as e:
            print(f"Error in caption generation: {str(e)}")
            return "Error: Failed to generate caption"

    def _get_image_key(self, dataset_name: str) -> str:
        """Get the image key from dataset info."""
        dataset_info = get_dataset_infos(dataset_name)["plain_text"]
        return next(key for key, feature in dataset_info.features.items() 
                   if str(feature).startswith("Image("))

    def _extract_captions(self, dataset: DatasetDict) -> List[str]:
        """Extract captions for all images in the dataset using LLaVA."""
        captions = []
        image_key = self._get_image_key(dataset.name)
        print(f"Using '{image_key}' as the image column")
        
        for i in tqdm(range(0, len(dataset), self.config.batch_size), desc="Generating captions"):
            batch = dataset[i:i + self.config.batch_size]
            
            try:
                caption = self._generate_caption(batch[image_key][0])  # Process one image at a time
                captions.append(caption)
            except Exception as e:
                print(f"Error processing image {i}: {e}")
                captions.append("Error: Failed to process image")
            
            time.sleep(0.1)
        
        return captions

    def _process_llm_query(self, text_input: str | List) -> str:
        """Process a text query with the Qwen GGUF model via llama-cpp-python."""
        try:
            # Format the prompt
            if isinstance(text_input, list):
                prompt_parts = []
                for msg in text_input:
                    if msg["role"] == "system":
                        prompt_parts.append(f"System: {msg['content']}")
                    elif msg["role"] == "user":
                        prompt_parts.append(f"User: {msg['content']}")
                prompt = "\n".join(prompt_parts) + "\nAssistant:"
            else:
                prompt = f"User: {text_input}\nAssistant:"
            
            # Generate response
            output = self.qwen_gguf(
                prompt=prompt,
                max_tokens=self.config.max_new_tokens,
                temperature=0.2,  # Low temperature for more consistent outputs
                top_p=0.9,
                top_k=20,
                stream=False
            )
            
            # Clean up response
            text_out = output["choices"][0]["text"].strip()
            
            # Extract only the relevant part (after any "Assistant:" prefix)
            if "Assistant:" in text_out:
                text_out = text_out.split("Assistant:", 1)[1].strip()
            
            return text_out
            
        except Exception as e:
            print(f"Error in query processing: {str(e)}")
            return ""

    def _discover_dimensions(self, captions: List[str]) -> List[str]:
        """Discover potential classification dimensions from captions using Qwen."""
        try:
            sample_size = min(10, len(captions))
            sample_captions = "\n".join(captions[:sample_size])
            main_subject = self._identify_main_subject(captions)
            
            initial_prompt = (
                f"Analyze these image captions and identify meaningful dimensions for categorizing the images:\n\n"
                f"{sample_captions}\n\n"
                f"What dimensions would you use to categorize these images? "
                f"Consider what aspects vary between the images.\n\n"
                f"Format your response exactly like this:\n"
                f"Dimensions: [list your discovered dimensions]"
            )
            
            initial_dimensions_text = self._process_llm_query([
                {"role": "system", "content": "You are analyzing image captions to discover natural categorization dimensions."},
                {"role": "user", "content": initial_prompt}
            ])
            
            # Extract dimensions from response
            if "Dimensions:" in initial_dimensions_text:
                dims = initial_dimensions_text.split("Dimensions:")[1].strip()
                dimensions = [d.strip().lower() for d in dims.split(",")]
            else:
                return []
            
            # Validate dimensions
            refined_dimensions = []
            for dim in dimensions:
                if len(dim.split()) <= 2:  # Allow up to two words for dimensions
                    refined_dimensions.append(dim)
            
            return refined_dimensions[:3]  # Keep top 3 dimensions
            
        except Exception as e:
            print(f"Error discovering dimensions: {str(e)}")
            return []

    def _identify_main_subject(self, captions: List[str]) -> str:
        """Identify the main subject from a set of captions."""
        try:
            prompt = (
                "What is the main subject or theme that these images have in common? "
                "Answer with a single word or short phrase.\n\n"
                f"Captions:\n{chr(10).join(captions[:3])}"
            )
            
            subject = self._process_llm_query([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ])
            
            return subject.strip().lower() or "general objects"
            
        except Exception as e:
            print(f"Error identifying main subject: {str(e)}")
            return "general objects"

    def _refine_dimensions(self, dimensions: List[str], captions: List[str]) -> Dict[str, List[str]]:
        """Discover categories for each dimension based on the captions."""
        refined_dims = {}
        sample_size = min(10, len(captions))
        
        for dimension in dimensions:
            try:
                sample_captions = "\n".join(captions[:sample_size])
                
                criteria_prompt = (
                    f"Based on these image captions:\n{sample_captions}\n\n"
                    f"What distinct categories do you observe for the dimension '{dimension}'?\n"
                    f"Consider how the images vary in terms of {dimension}.\n\n"
                    f"Format your response exactly like this:\n"
                    f"Categories: [list the categories you discovered]"
                )
                
                criteria_text = self._process_llm_query([
                    {"role": "system", "content": "You are discovering natural categories in image descriptions."},
                    {"role": "user", "content": criteria_prompt}
                ])
                
                # Extract categories
                if "Categories:" in criteria_text:
                    categories = criteria_text.split("Categories:")[1].strip()
                    category_list = [c.strip().lower() for c in categories.split(",")]
                    
                    # Validate categories
                    valid_categories = []
                    for cat in category_list:
                        if len(cat.split()) <= 2:  # Allow up to two words for categories
                            valid_categories.append(cat)
                    
                    refined_dims[dimension] = valid_categories if valid_categories else ["unknown"]
                else:
                    refined_dims[dimension] = ["unknown"]
                    
            except Exception as e:
                print(f"Error discovering categories for dimension {dimension}: {str(e)}")
                refined_dims[dimension] = ["unknown"]
        
        return refined_dims

    def _assign_subpopulations(self, captions: List[str], dimensions: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """Assign subpopulations based on discovered dimensions and categories."""
        assignments = []
        
        for caption in tqdm(captions, desc="Assigning subpopulations"):
            assignment = {}
            
            for dim, criteria in dimensions.items():
                try:
                    prompt = (
                        f"Caption: {caption}\n\n"
                        f"For the dimension '{dim}', which category best describes this image?\n"
                        f"Available categories: {', '.join(criteria)}\n\n"
                        f"Reply with just the category name."
                    )
                    
                    category = self._process_llm_query([
                        {"role": "system", "content": "You are categorizing image descriptions."},
                        {"role": "user", "content": prompt}
                    ])
                    
                    # Clean up and validate response
                    category = category.strip().lower()
                    if category not in [c.lower() for c in criteria]:
                        category = "unknown"
                    
                    assignment[dim] = category
                    
                except Exception as e:
                    print(f"Error assigning category for dimension {dim}: {str(e)}")
                    assignment[dim] = "unknown"
            
            assignments.append(assignment)
        
        return assignments

    def visualize_subpopulations(self, results: Dict[str, Any]):
        """Generate visualizations for subpopulation analysis."""
        print("\n=== Dataset Analysis Results ===\n")
        
        # 1. Show sample of captions
        print("📸 Sample Captions:")
        for i, caption in enumerate(results["captions"][:5], 1):  # Show first 5 captions
            print(f"  {i}. {caption}")
        
        if len(results["captions"]) > 5:
            print(f"  ... and {len(results["captions"]) - 5} more")
        
        # 2. Show dimensions and their distributions
        print("\n🎯 Discovered Dimensions and Categories:")
        df = pd.DataFrame(results["subpopulations"])
        
        for dim, categories in results["dimensions"].items():
            print(f"\n  {dim.title()}:")
            counts = df[dim].value_counts()
            total = len(df)
            
            print("  Categories:")
            for category in categories:
                count = counts.get(category, 0)
                percentage = (count / total) * 100
                print(f"    • {category}: {count} images ({percentage:.1f}%)")
        
        # 3. Show interesting patterns
        print("\n📊 Key Insights:")
        for dim in results["dimensions"].keys():
            most_common = df[dim].mode().iloc[0] if not df[dim].empty else "unknown"
            coverage = (df[dim] != "unknown").mean() * 100
            print(f"  • {dim.title()}:")
            print(f"    - Most common category: {most_common}")
            print(f"    - Classification success rate: {coverage:.1f}%")
        
        print("\n=== End of Analysis ===\n")
        
        return df

    def process_dataset(self, dataset: DatasetDict) -> Dict[str, Any]:
        """
        Process a dataset through the complete pipeline.
        
        Args:
            dataset: HuggingFace dataset containing images
            
        Returns:
            Dictionary containing subpopulation assignments and metadata
        """
        # 1. Extract captions via LLaVA
        captions = self._extract_captions(dataset)
        
        # 2. Discover dimensions using Qwen GGUF
        dimensions = self._discover_dimensions(captions)
        
        # 3. Refine dimensions and criteria using Qwen GGUF
        refined_dims = self._refine_dimensions(dimensions, captions)
        
        # 4. Assign subpopulations using Qwen GGUF
        assignments = self._assign_subpopulations(captions, refined_dims)
        
        return {
            "subpopulations": assignments,
            "dimensions": refined_dims,
            "captions": captions
        }
