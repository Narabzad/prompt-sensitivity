#!/bin/bash

echo "Starting baseline evaluations..."

# Create results directory if it doesn't exist
mkdir -p results

# Step 1: Train and evaluate text classification baselines
echo "Running text classification baselines..."
python baselines/text_classification/binary_text_classification_train.py
python baselines/text_classification/binary_text_classification_eval.py

# Step 2: Run specificity-based QPP baselines
echo "Running specificity-based QPP baselines..."
python baselines/specifity-based_QPP/specifity-based_QPP_baseline_results_script.py

# Step 3: Run supervised QPP baselines
echo "Running supervised QPP baselines..."
python baselines/supervised_QPP/supervised_QPP_baseline_results_script.py

echo "All baseline evaluations completed. Results stored in results/"
