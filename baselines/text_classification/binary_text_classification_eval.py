import pandas as pd
import logging
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import json

import os
import glob
from sklearn.metrics import precision_score, recall_score

import os



current_directory = os.getcwd()



# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO for general messages or DEBUG for detailed debugging
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("eval_execution_log.log"),  # Log to a file named 'execution_log.log'
        logging.StreamHandler()  # Optionally log to console
    ]
)


def get_latest_file(directory):
    files = glob.glob(f"{directory}/*")  # Get all files in directory
    latest_file = max(files, key=os.path.getctime)  # Get file with latest creation time
    return latest_file


def prompt_em_pair_from_jsonl_to_df(file_path):

    dataset_exact_matches_raw = open(file_path).readlines()
    # Parsing each line as JSON and extracting the desired information
    dataset_exact_matches = [json.loads(item) for item in dataset_exact_matches_raw]
    result_dataset = [[item["prompt"], item["exact_match"]] for item in dataset_exact_matches]
    df = pd.DataFrame(result_dataset)
    df.columns = ["text", "labels"]

    return df




model_args = ClassificationArgs(use_cached_eval_features=True)


datasets = ["hotpotqa", "triviaqa"] #hotpotqa
generative_models = ["llama3_8B", "mistral"] #mistral


classifier_models = ["bert-base-uncased"]
classifier_model_types = ["bert"]


for dataset in datasets:
    for gen_model in generative_models:
        train_file_path = current_directory + f"/../../prompt_set/{dataset}/{dataset}_{gen_model}_dataset_train.jsonl"

        test_file_path = current_directory + f"/../../prompt_set/{dataset}/{dataset}_{gen_model}_dataset_test.jsonl"

        train_df = prompt_em_pair_from_jsonl_to_df(train_file_path)
        eval_df = prompt_em_pair_from_jsonl_to_df(test_file_path)

        for i, model in enumerate(classifier_models):
            # read the trained models

            if "/" in model:
                model_name = model.split("/")[-1]
            else:
                model_name = model

            
            read_model = ClassificationModel(classifier_model_types[i], 
                        get_latest_file(current_directory + f'/{dataset}_{gen_model}_{model_name}_training')
                    )
            # Evaluate the model
            result, model_outputs, wrong_predictions = read_model.eval_model(eval_df, output_dir=current_directory + f'/results/{dataset}_{gen_model}_{model_name}_evaluation', recall=recall_score, precision=precision_score)



            # logging.info(f"model:\n{result}\n\nmodel_outputs:\n{model_outputs}\n\nwrong_predictions:\n{wrong_predictions}")


