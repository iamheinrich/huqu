from huqu.stages.dataset_loading import DatasetLoadingStage
from huqu.stages.caption_generation import CaptionGenerationStage
from huqu.stages.criteria_initilization import CriteriaInitializationStage
from huqu.stages.criteria_refinement import CriteriaRefinementStage
from huqu.stages.image_assignment import ImageAssignmentStage
from huqu.models.gemini import GeminiModel
from huqu.models.groq import GroqVisionModel
from huqu.models.chatgpt import GPT4oMiniMLLM, GPT4oMiniLLM

import pandas as pd
import json

print("#"*80)
print('Starting dataset loading')
print("#"*80)

dataset_loader = DatasetLoadingStage()
kwargs = {'trust_remote_code': True} 
dataset = dataset_loader.process(**kwargs)

assert len(dataset) == 150

print("#"*80)
print('Finished loading dataset')
print("#"*80)


print("#"*80)
print('Starting caption generation')
print("#"*80)   

mllm = GPT4oMiniMLLM()
caption_generator = CaptionGenerationStage(mllm)
captions = caption_generator.process(dataset)

print("#"*80)
print('Finished caption generation')
print("#"*80)   

print("#"*80)
print('Starting criteria initialization')
print("#"*80)   

df = pd.read_parquet("data/captions.parquet")

llm = GPT4oMiniLLM()
criteria_initializer = CriteriaInitializationStage(llm)
criteria = criteria_initializer.process()

# Save criteria to JSON for inspection
with open("data/unrefined_criteria.json", "w") as f:
    json.dump({
        "dimensions": criteria.dimensions,
        "attributes": criteria.attributes
    }, f, indent=2)

print(f"Saved classification criteria to data/unrefined_criteria.json")

unrefined_criteria = json.load(open("data/unrefined_criteria.json"))

print("#"*80)
print('Finished criteria initialization')
print("#"*80)   

print("#"*80)
print('Starting criteria refinement')
print("#"*80)   

criteria_refiner = CriteriaRefinementStage(llm)
criteria = criteria_refiner.process(unrefined_criteria)

# Save refined criteria to JSON
with open("data/refined_criteria.json", "w") as f:
    json.dump(criteria, f, indent=2)

print("#"*80)
print('Finished criteria refinement')
print("#"*80)   

print("#"*80)
print('Starting image assignment')
print("#"*80)   

refined_criteria = json.load(open("data/refined_criteria.json"))
image_assignment = ImageAssignmentStage(llm)
final_ssd = image_assignment.process(refined_criteria)

print("#"*80)
print('Finished image assignment')
print("#"*80)   

print('Pipeline finished running')