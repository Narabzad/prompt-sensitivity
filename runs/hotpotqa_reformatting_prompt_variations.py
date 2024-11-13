import json

from sklearn.metrics import f1_score as sklearn_f1_score
import numpy as np
import time
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO for general messages or DEBUG for detailed debugging
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("execution_log_formatting_prompt_variations.log"),  # Log to a file named 'execution_log.log'
        logging.StreamHandler()  # Optionally log to console
    ]
)


import json
import uuid

import os



current_directory = os.getcwd()



desired_number_of_variations = 10


def transform_jsonl(file_path):
    transformed_data = []
    
    with open(file_path, 'r') as file:
        number_of_bad_data = 0

        for line in file:
            data = json.loads(line)
            
            # Generating unique IDs
            original_id = f"hpqa_{str(uuid.uuid4())[:8]}"
            reference_prompt_id = original_id

            number_of_variations = 0

            transformed_data.append({
                "prompt_id": original_id,
                "reference_prompt_id": reference_prompt_id,
                "prompt": data['original_prompt'],
                "expected_answer": [data['expected_answer']],
                # "model": model_mapping[var_model_name],
                # "llm_response": response,
                "model_answers": data['model_answers']
            })


            for index, (model_name, variations) in enumerate(data["model_variations"].items()):
                # prompt_id = f"alt_{original_id}_{index+1}"
                reference_prompt_id = original_id

                # Create transformed records for variations
                for var_index, variation in enumerate(variations.values()):
                    var_prompt_id = f"alt_{original_id}_{var_index+1}"

                    transformed_data.append({
                        "prompt_id": var_prompt_id,
                        "reference_prompt_id": reference_prompt_id,
                        "prompt": variation['generated_prompt'],
                        "expected_answer": [data['expected_answer']],
                        "model_answers": variation['model_answers'],
                    })
                    number_of_variations = number_of_variations + 1
                if number_of_variations != desired_number_of_variations:
                    logging.info(f"data does not have {desired_number_of_variations}, has {number_of_variations} number of variations, the data looks as follows:\n{data}")
                    number_of_bad_data = number_of_bad_data + 1

    logging.info(f"\n\nnumber of bad data = {number_of_bad_data}")
    return transformed_data


def map_fields(init_dict, map_dict, res_dict=None):
    res_dict = res_dict or {}
    for k, v in init_dict.items():
        # print("Key: ", k)
        if isinstance(v, dict):
            # print("value is a dict - recursing")
            v = map_fields(v, map_dict)
        elif k in map_dict.keys():
            # print("Remapping:", k, str(map_dict[k]))
            k = str(map_dict[k])
        res_dict[k] = v
    return res_dict



model_mapping = {"llama3.1:8b" : "llama3_8B", "mistral-nemo:latest" : "mistral-nemo"}

# Example usage
transformed_data = transform_jsonl(current_directory + '/generated_datasets/hotpotqa_prompt_variations_unformatted.jsonl')

# Save to a new JSONL file
with open(current_directory + '/generated_datasets/hotpotqa_prompt_variations.jsonl', 'w') as outfile:
    for record in transformed_data:
        json.dump(record, outfile)
        outfile.write('\n')
