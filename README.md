# PromptSET: A Dataset for Evaluating Prompt Sensitivity in Large Language Models

## Summary
PromptSET is a comprehensive dataset designed to investigate the effects of prompt variations on Large Language Model (LLM) performance. Built upon TriviaQA and HotpotQA datasets, PromptSET enables researchers to study how slight modifications in prompt formulation can significantly impact LLM responses.

|Description|Details|
|---|---|
|Collection Source|TriviaQA and HotpotQA datasets|
|Number of Original Prompts|11,469|
|Variations per Prompt|9|
|Total Prompt Variations|114,690|
|Models Evaluated|LLaMA 3.1 (8B), Mistral-nemo|

## Dataset Structure

### Main Components
- `prompt_set/`
  - `triviaqa/`
    - `triviaqa_llama3_8B_dataset_train.jsonl`
    - `triviaqa_llama3_8B_dataset_test.jsonl`
    - `triviaqa_mistral_dataset_train.jsonl`
    - `triviaqa_mistral_dataset_test.jsonl`
  - `hotpotqa/`
    - `hotpotqa_llama3_8B_dataset_train.jsonl`
    - `hotpotqa_llama3_8B_dataset_test.jsonl`
    - `hotpotqa_mistral_dataset_train.jsonl`
    - `hotpotqa_mistral_dataset_test.jsonl`

### Data Format
Each entry in the JSONL files contains:
```json
{
    "prompt_id": "unique_identifier",
    "reference_prompt_id": "original_prompt_id",
    "prompt": "prompt_text",
    "expected_answer": ["correct_answer"],
    "model_answers": {
        "llama3_8B": "model_response",
        "mistral-nemo": "model_response"
    }
}
```

## Generating Prompt Variations

### Requirements
- Python 3.9+
- Ollama
- Required Python packages (specified in requirements.txt)

### Process
1. Data extraction from HotpotQA/TriviaQA
2. Generation of prompt variations using LLMs
3. Answer generation for each variation
4. Formatting and storing results

To generate prompt variations, run:
```bash
./generate_variations.sh
```

## Running the Baselines

### Available Baselines

1. **LLM-Based Evaluation**
- Direct use of LLMs (LLaMA and Mistral) for self-assessment
- Evaluation of model's ability to predict its own performance

2. **Query Performance Prediction (QPP)**
- Pre-retrieval QPP methods including:
  - Closeness Centrality (CC)
  - Degree Centrality (DC)
  - Inverse Edge Frequency (IEF)
  - PageRank
- BERT-PE: A supervised pre-retrieval QPP model
- Implementation in `baselines/specifity-based_QPP/specifity-based_QPP_baseline_results_script.py` and `baselines/supervised_QPP/supervised_QPP_baseline_results_script.py`

3. **Text Classification**
- BERT-based classification model for predicting prompt effectiveness
- Implementation in `baselines/text_classification/binary_text_classification_train.py` and `baselines/text_classification/binary_text_classification_eval.py`

To run all baselines:
```bash
./run_baselines.sh
```

## Key Findings

1. When the original prompt is answered correctly by an LLM, there is a higher likelihood of variations also yielding correct answers. Conversely, if the original prompt is answered incorrectly, most variations tend to fail as well.

2. Variations that are more similar to the original prompt (based on embedding similarity) are more likely to generate both correct responses and accurate predictions of answerability, suggesting a potential bias toward training data patterns.

3. Prompt reformulation can enhance LLM effectiveness - in cases where both LLMs failed to answer the original prompt correctly, certain variations enabled successful responses from one or both models.

4. Among baseline methods:
   - Specificity-based QPP methods show relatively weak performance on both datasets
   - BERT-PE demonstrates competitive performance, particularly on HotpotQA
   - LLM self-evaluation shows reasonable performance on TriviaQA but lacks consistency on HotpotQA

## License


## Contact
