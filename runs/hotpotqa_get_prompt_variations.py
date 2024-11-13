import json
import ollama
from ollama import ps, pull, chat

from sklearn.metrics import f1_score as sklearn_f1_score
import numpy as np
import time
import logging

import openlit
openlit.init(otlp_endpoint="http://127.0.0.1:4318", )


import os



current_directory = os.getcwd()



# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO for general messages or DEBUG for detailed debugging
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("execution_log.log"),  # Log to a file named 'execution_log.log'
        logging.StreamHandler()  # Optionally log to console
    ]
)

# Suppress logs from HTTPX if OpenAI uses it
logging.getLogger("httpx").setLevel(logging.CRITICAL)

# Suppress the 'http' logger used by some libraries
logging.getLogger("http").setLevel(logging.CRITICAL)

def extract_exact_matches(dataset, output_file="prompt_answers_models_filtered.jsonl"):
    # Open the file in write mode
    prompt_id = 0
    with open(output_file, 'w') as file:
        for item in dataset:
            expected_answer = item['expected_answer']
            model_answers = item['model_answers']
            
            # Check if all model answers match the expected answer
            if all(answer == expected_answer for answer in model_answers.values()):
                # Write the matching data to the jsonl file
                item['prompt_id'] = prompt_id
                json.dump(item, file)
                file.write('\n')

                prompt_id = prompt_id + 1


# Function to compute Exact Match
def exact_match(pred, true):
    return int(pred.strip().lower() == true.strip().lower())


# Token-level F1 score computation using scikit-learn's f1_score
def f1_score(pred, true):
    # Split the predicted and true answers into tokens
    pred_tokens = pred.strip().lower().split()
    true_tokens = true.strip().lower().split()
    
    # Create a list of common tokens
    common = set(pred_tokens) & set(true_tokens)
    num_same = len(common)
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(true_tokens)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def compute_scores(jsonl_file_path, start_idx=-1, end_idx=-1):
    # Load the dataset
    dataset_raw = open(jsonl_file_path).readlines()
    dataset = [json.loads(item) for item in dataset_raw]

    if start_idx != -1 and end_idx != -1:
        dataset = dataset[start_idx:end_idx]
    elif start_idx != -1:
        end_idx = len(dataset)
        dataset = dataset[start_idx:]
    elif end_idx != -1:
        start_idx = 0
        dataset = dataset[:end_idx]
    else:
        start_idx = 0
        end_idx = len(dataset)
    
    # Initialize dictionary to store scores for each model
    model_scores = {}

    # Loop through each entry in the dataset
    for entry in dataset:
        expected_answer = entry["expected_answer"]
        model_answers = entry["model_answers"]
        
        # For each model's prediction, compute the scores
        for model, answer in model_answers.items():
            if model not in model_scores:
                model_scores[model] = {'f1': [], 'em': []}
            
            em = exact_match(answer, expected_answer)
            f1 = f1_score(answer, expected_answer)
            
            model_scores[model]['f1'].append(f1)
            model_scores[model]['em'].append(em)

    # Calculate the average F1 and Exact Match scores for each model
    print(f"Computing scores for entries {start_idx} to {end_idx}")
    for model in model_scores:
        avg_f1 = np.mean(model_scores[model]['f1'])
        avg_em = np.mean(model_scores[model]['em'])
        print(f"Model: {model}")
        print(f"  Average F1 Score: {avg_f1:.2f}")
        print(f"  Exact Match Score: {avg_em:.2f}")


def add_prompt_id(input_path, output_path):
    dataset_raw = open(input_path).readlines()
    dataset = [json.loads(item) for item in dataset_raw]
    
    for ind, i  in enumerate(dataset):
        i['prompt_id'] = ind
        with open(output_path, 'a') as file:
            file.write(json.dumps(i) + '\n')

def get_model_answers(user_prompt, models):
    model_answers = dict()
    qa_system_prompt = "You are a professional question-answering system. Return just the answer based on the given context. Do not tell anything else. Just the answer under 5 words. Do not include any data"
    # print("\noriginal prompt: " + user_prompt)
    for m in models:
        prompt_answer = ollama.generate(model=m, prompt=user_prompt, system=qa_system_prompt, options={"temperature":1, "seed":42})['response']
        # time.sleep(1)
        # print(f"Model: {m:<22}" + prompt_answer)
        model_answers[m] = prompt_answer
    return model_answers

def generate_prompt_variations(user_prompt, models_to_reformulate, models_to_answer, num_variations=5):
    model_variations = dict()
    reformulator_system_prompt = f"You are a prompt engineer who can develop {num_variations} prompts from a given question. Do not answer the question. Your task is to just reformulate the question and create {num_variations} different versions of it that preserve the original information need. Only return the {num_variations} reformulated prompts related to the original question, separated by semicolons. Do not include anything else in your response."
    
    for model in models_to_reformulate:
        model_variations[model] = dict()
        reformulated_prompts = ollama.generate(model=model, prompt=user_prompt, system=reformulator_system_prompt, options={"temperature":1, "seed":42})['response']
        # time.sleep(1)
        for i, gen_p in enumerate(reformulated_prompts.split(';')):
            model_variations[model]["variation_" + str(i+1)] = dict()
            model_variations[model]["variation_" + str(i+1)]['generated_prompt'] = gen_p.strip()
            model_variations[model]["variation_" + str(i+1)]['model_answers'] = get_model_answers(gen_p.strip(), models_to_answer)
    return model_variations


def data_extractor_hotpotqa(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        extracted_data = []
        for record in data:
            record_attributes = {'question': record['question'], 'answer': record['answer'], 'level': record['level']}
            extracted_data.append(record_attributes)
        return extracted_data


# def get_prompt_answers_models(user_prompt, models):
#     qa_system_prompt = "You are a professional question-answering system. Return just the answer based on the given context. Do not tell anything else. Just the answer under 5 words. Do not include any data"
#     # print("\noriginal prompt: " + user_prompt)
#     prompt_answers_models = {'prompt': user_prompt, 'model_answers': {}}
#     for m in models:
#         prompt_answer = ollama.generate(model=m, prompt=user_prompt, system=qa_system_prompt, options={"temperature":1, "seed":42})['response']
#         # print(f"Model: {m:<22}" + prompt_answer)
#         prompt_answers_models['model_answers'][m] = prompt_answer
#     return prompt_answers_models

# ROADMAP: extract exact matches --> run the prompt on different models to get variations --> get answers to those prompts --> evaluate the answers



# jsonl_file_path = f"/mnt/data/arazavi/test_on_llama/prompt_answers_models.jsonl"
# # models = ["llama3.1:70b", "phi3:14b", "mistral-nemo:latest"]
# models = ["llama3.1:8b", "mistral-nemo:latest"]



# # read the raw data
# dataset_raw = open(jsonl_file_path).readlines()
# dataset = [json.loads(item) for item in dataset_raw]

# # extract exact matches
# exact_matches_output_path = "/mnt/data/arazavi/test_on_llama/prompt_answers_models_filtered.jsonl"
# extract_exact_matches(dataset, output_file=exact_matches_output_path)


# dataset_exact_matches_raw = open(exact_matches_output_path).readlines()
# dataset_exact_matches = [json.loads(item) for item in dataset_exact_matches_raw]


hotpotqa_file_path = current_directory + '/../collection/hotpot_train_v1.1.json'
extracted_data = data_extractor_hotpotqa(hotpotqa_file_path)

models_to_answer = ['llama3.1:8b', 'mistral-nemo:latest']
models_to_reformulate = ['llama3.1:8b']

# generate prompt variations
start_time = time.time()
for i, record in enumerate(extracted_data):
# for elem in dataset_exact_matches:
    new_json_line = dict()
    new_json_line['original_prompt'] = record['question']
    new_json_line['hotpotqa_level'] = record['level']
    new_json_line['expected_answer'] = record['answer']
    new_json_line['model_answers'] = get_model_answers(record['question'], models_to_answer)
    new_json_line['model_variations'] = generate_prompt_variations(record['question'], models_to_reformulate, models_to_answer, num_variations=10)
    with open(current_directory + '/generated_datasets/hotpotqa_prompt_variations_unformatted.jsonl', 'a') as file:
            file.write(json.dumps(new_json_line) + '\n')
    if i % 500 == 0:
        elapsed_time = time.time()-start_time
        print(f"Processed {i + 1} prompts. Time taken since last check: {elapsed_time:.2f} seconds")
        logging.info(f"Processed {i + 1} prompts. Time taken since last check: {elapsed_time:.2f} seconds")
        start_time = time.time()
