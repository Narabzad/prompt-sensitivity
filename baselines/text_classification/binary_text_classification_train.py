import pandas as pd
import logging
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import json
import os
from sklearn.metrics import precision_score, recall_score

import os



current_directory = os.getcwd()



# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO for general messages or DEBUG for detailed debugging
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train_execution_log.log"),  # Log to a file named 'execution_log.log'
        logging.StreamHandler()  # Optionally log to console
    ]
)


def prompt_em_pair_from_jsonl_to_df(file_path):

    dataset_exact_matches_raw = open(file_path).readlines()
    # Parsing each line as JSON and extracting the desired information
    dataset_exact_matches = [json.loads(item) for item in dataset_exact_matches_raw]
    result_dataset = [[item["prompt"], item["exact_match"]] for item in dataset_exact_matches]
    df = pd.DataFrame(result_dataset)
    df.columns = ["text", "labels"]

    return df


def rename_directory(old_path, new_path):
    if os.path.isdir(old_path):  # Check if the directory exists
        os.rename(old_path, new_path)
        print(f"Directory renamed from '{old_path}' to '{new_path}'.")
    else:
        print(f"Directory '{old_path}' does not exist, so it was not renamed.")


def training_model(train_df, model, output_dir, recalls, precisions):
    # Train the model
    model.train_model(train_df, output_dir=output_dir, recall=recalls, precision=precisions)


# Optional model configuration
model_args = ClassificationArgs()

# # Create a ClassificationModel
# model = ClassificationModel(
#     "roberta", "roberta-base", args=model_args
# )




datasets = ["hotpotqa", "triviaqa"] #triviaqa
generative_models = ["llama3_8B", "mistral"] #mistral
# classifier_models = [
#     ClassificationModel("roberta", "roberta-base", args=model_args), \
#     ClassificationModel("bert", "bert-base-uncased", args=model_args),\
#     ClassificationModel("deberta", "microsoft/deberta-base", args=model_args)
#     ]


for dataset in datasets:
    for gen_model in generative_models:
        train_file_path = current_directory + f"/../../prompt_set/{dataset}/{dataset}_{gen_model}_dataset_train.jsonl"

        test_file_path = current_directory + f"/../../prompt_set/{dataset}/{dataset}_{gen_model}_dataset_test.jsonl"

        train_df = prompt_em_pair_from_jsonl_to_df(train_file_path)
        eval_df = prompt_em_pair_from_jsonl_to_df(test_file_path)


        classifier_model = ClassificationModel("bert", "bert-base-uncased", args=model_args)
        model_name = classifier_model.args.model_name
        output_dir=current_directory + f'/{dataset}_{gen_model}_{model_name}_training'
        training_model(train_df, classifier_model, output_dir, recalls=recall_score, precisions=precision_score)
        print(f"\nModel trained completely: {model_name}")
        rename_directory(current_directory + '/outputs', \
        current_directory + f'/outputs_{dataset}_{gen_model}_{model_name}_training')

        # for model in classifier_models:
        #     # Train the model
        #     if "/" in model.args.model_name:
        #         model_name = model.args.model_name.split("/")[-1]
        #     else:
        #         model_name = model.args.model_name
        #     model.train_model(train_df, output_dir=current_directory + f'/{dataset}_{gen_model}_{model_name}_training', recalloo=recall_score, precisionoo=precision_score)

        #     print(f"\nModel trained completely: {model_name}")

        #     rename_directory(current_directory + '/outputs', \
        #     current_directory + f'/outputs_{dataset}_{gen_model}_{model_name}_training')
