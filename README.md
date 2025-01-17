# Dataset Statistics Analyzer Prototype

## Overview

This project is a prototype for a dataset statistics analyzer focused on Hugging Face image datasets. The goal is to:

1. Accept a Hugging Face dataset path as input.
2. Enable users to specify whether the dataset contains humans and, if so, choose from a predefined set of protected attributes (e.g., gender, ethnicity).
3. Tag images in the dataset using CLIP and predefined prompts for each attribute.
4. Compute basic image statistics (e.g., brightness, hue).
5. Detect potential image issues using Cleanlab.
6. Provide a summary of the analysis.

## Architecture Design

### Core Workflow

1. **Input Layer**:

   - Accepts:
     - Dataset path (Hugging Face only in this prototype).
     - Option to specify whether fairness analysis is required.
   - If fairness analysis is enabled:
     - Expects datasets to contain humans.
     - Checks the Hugging Face dataset statistics to verify that the dataset contains classes.
     - Requires the user to specify the classes that contain humans.
     - Allows the user to choose protected attributes for analysis (e.g., gender, ethnicity).

2. **Tagging Service**:

   - Utilizes CLIP for image tagging using predefined prompts.
   - Processes images using binary prompt pairs (e.g., "This photo contains a man" / "This photo does not contain a man").
   - Outputs probabilistic logits.

3. **Image Statistics Service**:

   - Computes basic aggregates:
     - Brightness
     - Hue
     - ...

4. **Issue Detection Service**:

   - Leverages Cleanlab to identify noisy or problematic data points.

5. **Data Storage**:

   - Uses a structured format (e.g., json to store analysis results.
   - Example structure:
     ```json
     {
         "image_id_1": {
             "tags": {
                 "gender": {
                     "male": true,
                     "female": false
                 },
                 "ethnicity": {
                     "white": true,
                     "hispanic": false
                 }
             },
             "stats": {"brightness": 0.7, "hue": 0.5},
             "issues": {"label_noise": true}
         },
         ...
     }
     ```

6. **API Layer**:

   - Provides user-facing methods for result exploration:
     - `show_fairness_analysis()`
     - `show_overview()`

### Prototype Scope

- **Attributes**:

  - Initial protected attributes: Gender, Ethnicity, Age, Weight.

- **Image Statistics**:

  - Computes basic aggregates (brightness, hue, etc).

- **Datasets**:

  - Supports Hugging Face datasets only.

- **Output**:

  - Terminal-based summary of analysis results.

## Data Structures

- **Analysis Results**:

  - Nested dictionaries or JSON-like structures for flexibility:
    ```python
    {
        "image_id": {
            "tags": {
                "gender": {
                    "male": true,
                    "female": false
                },
                "ethnicity": {
                    "white": true,
                    "hispanic": false
                }
            },
            "stats": {"brightness": 0.7, "hue": 0.5},
            "issues": {"label_noise": true}
        }
    }
    ```

- **Temporary Data**:

  - Use Pandas DataFrames for intermediate calculations and efficient querying.

## Best Practices

1. **Modular Design**:

   - Separate functionality for tagging, statistics, and issue detection into distinct modules.

2. **Ease of Use**:

   - Maintain simple and clear APIs.
   - Provide defaults and sensible error messages for user inputs.

## Implementation Plan

1. **Tagging Service**:

   - Integrate CLIP from Hugging Face.
   - Implement binary prompt evaluation for gender and ethnicity.

2. **Image Statistics**:

   - Calculate brightness, hue etc using PIL.

3. **Issue Detection**:

   - Use Cleanlab for detecting image issues.

4. **API Development**:

   - Build basic methods to display analysis results.

5. **Testing and Validation**:

   - Test the prototype on a couple of Hugging Face datasets.

## Deliverables

- A Python prototype with:
  - Dataset loading and tagging capabilities.
  - Basic image statistics computation.
  - Noise detection integration with Cleanlab.
  - Terminal-based summary methods.

---

This document outlines the simplified structure and scope for the prototype while keeping future extensibility in mind. Let me know if further adjustments are needed!
