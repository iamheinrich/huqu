# HuQu - Hugging Face Dataset Quality Assessment

A library to find hidden subpopulations in Image Classification Datasets from Hugging Face.

## Description

HuQu is a Python library designed to automatically identify and analyze hidden subpopulations in image classification datasets from Hugging Face. By leveraging Large Language Models (LLMs), it enables systematic dataset auditing before model training, helping to uncover potential biases and dataset artifacts that could affect model performance.

Key features:

- Automatic subpopulation discovery using LLMs without manual annotation
- Intraclass analysis to identify subgroup imbalances within individual classes
- Interclass analysis to detect cross-class artifacts and potential shortcut learning risks
- Configurable thresholds for bias detection
- Structured output for actionable dataset insights

The library helps machine learning practitioners to:

- Identify underrepresented and overrepresented subgroups automatically without additional effort
- Detect potential sources of bias before model training
- Guide dataset augmentation and refinement strategies for improving model fairness and robustness

## Requirements

- Python 3.12+
- Poetry for dependency management

### Main Dependencies

- datasets
- torch
- numpy
- pandas
- huggingface-hub
- google-genai
- openai
- python-dotenv
- pillow

## Installation

1. Clone the repository:

```bash
git clone https://github.com/iamheinrich/huqu.git
```

2. Install dependencies using Poetry:

```bash
poetry install
```

3. Set up environment variables:
   Create a `.env` file in the root directory with the following content:

```env
OPENAI_API_KEY=your_openai_api_key
...
```

## Project Structure

- `huqu/`: Main library code
- `scripts/`: Utility scripts and tools
- `pipeline_config.yaml`: Configuration for the evaluation pipeline

## Usage

[Basic usage instructions will go here]

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- Hendrik Schulze Broering
- Katrin Lenzeder
