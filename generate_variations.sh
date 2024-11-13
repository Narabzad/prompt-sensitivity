#!/bin/bash

echo "Starting prompt variation generation process..."

# Create necessary directories if they don't exist
mkdir -p generated_datasets

# Step 1: Generate initial prompt variations
python runs/hotpotqa_get_prompt_variations.py

# Step 2: Reformat the generated variations
python runs/hotpotqa_reformatting_prompt_variations.py

echo "Prompt variation generation completed. Results stored in generated_datasets/"
