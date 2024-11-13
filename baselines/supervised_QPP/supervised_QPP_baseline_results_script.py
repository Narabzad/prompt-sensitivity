import pandas as pd
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os



current_directory = os.getcwd()



def calculate_metrics(df):
    accuracy = accuracy_score(df['exact_match'], df['predicted_label'])
    precision = precision_score(df['exact_match'], df['predicted_label'])
    recall = recall_score(df['exact_match'], df['predicted_label'])
    f1 = f1_score(df['exact_match'], df['predicted_label'])
    return accuracy, precision, recall, f1


# Read the JSONL file
triviaqa_test_llama_jsonl = current_directory + '/../../prompt_set/triviaqa/triviaqa_llama3_8B_dataset_test.jsonl'
triviaqa_test_mistral_jsonl = current_directory + '/../../prompt_set/triviaqa/triviaqa_mistral_dataset_test.jsonl'

hotpotqa_test_llama_jsonl = current_directory + '/../../prompt_set/hotpotqa/hotpotqa_llama3_8B_dataset_test.jsonl'
hotpotqa_test_mistral_jsonl = current_directory + '/../../prompt_set/hotpotqa/hotpotqa_mistral_dataset_test.jsonl'

# Extract only the required fields
data = []
with open(triviaqa_test_llama_jsonl, 'r') as f:
    for line in f:
        record = json.loads(line)
        # Append only the selected fields to the data list
        data.append({
            "prompt_id": record.get("prompt_id"),
            "exact_match": record.get("exact_match")
        })

df_trivia_llama_id_em = pd.DataFrame(data)



# Extract only the required fields
data = []
with open(triviaqa_test_mistral_jsonl, 'r') as f:
    for line in f:
        record = json.loads(line)
        # Append only the selected fields to the data list
        data.append({
            "prompt_id": record.get("prompt_id"),
            "exact_match": record.get("exact_match")
        })

df_trivia_mistral_id_em = pd.DataFrame(data)


# Extract only the required fields
data = []
with open(hotpotqa_test_llama_jsonl, 'r') as f:
    for line in f:
        record = json.loads(line)
        # Append only the selected fields to the data list
        data.append({
            "prompt_id": record.get("prompt_id"),
            "exact_match": record.get("exact_match")
        })

df_hotpot_llama_id_em = pd.DataFrame(data)



# Extract only the required fields
data = []
with open(hotpotqa_test_mistral_jsonl, 'r') as f:
    for line in f:
        record = json.loads(line)
        # Append only the selected fields to the data list
        data.append({
            "prompt_id": record.get("prompt_id"),
            "exact_match": record.get("exact_match")
        })

df_hotpot_mistral_id_em = pd.DataFrame(data)


qpps = ["triviaqa_BertPE.txt",
        "hotpotqa_BertPE.txt"]




for qpp in qpps:
    # Read the file into a DataFrame
    column_name = qpp.split("/")[-1].split(".")[0].split("_")[-1]
    # df = pd.read_csv(qpp, delim_whitespace=True, header=None, names=["prompt_id", column_name])
    data = []
    with open(qpp, "r") as file:
        for line in file:
            # Split each line by whitespace and convert to a list of values
            row = line.split()
            data.append(row)

    # Convert the data into a DataFrame
    df = pd.DataFrame(data)
    df.columns = ["prompt_id", column_name]
    # print(df.head())
    # print(df.info())
    df[column_name] = df[column_name].astype(float)
    # Calculate the average of the 'page_rank' column
    average_metric = df[column_name].mean()

    df['predicted_label'] = df[column_name].apply(lambda x: 1 if x > average_metric else 0)
    
    


    if "triviaqa" in qpp:
        merged_df_trivia_llama = pd.merge(df, df_trivia_llama_id_em, on="prompt_id", how="inner")
        merged_df_trivia_mistral = pd.merge(df, df_trivia_mistral_id_em, on="prompt_id", how="inner")

        # Assuming qpp and merged_df_llama are defined earlier in the code
        file_name = qpp.split("/")[-1].split(".")[0]
        file_path = f"results/{file_name}_llama.txt"
        with open(file_path, 'w') as f:
            accuracy, precision, recall, f1 = calculate_metrics(merged_df_trivia_llama)
            f.write(json.dumps({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }) + "\n")
            

        file_path = f"results/{file_name}_mistral.txt"
        with open(file_path, 'w') as f:
            accuracy, precision, recall, f1 = calculate_metrics(merged_df_trivia_mistral)
            f.write(json.dumps({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }) + "\n")

    else:
        merged_df_hotpot_llama = pd.merge(df, df_hotpot_llama_id_em, on="prompt_id", how="inner")
        merged_df_hotpot_mistral = pd.merge(df, df_hotpot_mistral_id_em, on="prompt_id", how="inner")

        # Assuming qpp and merged_df_llama are defined earlier in the code
        file_name = qpp.split("/")[-1].split(".")[0]
        file_path = f"results/{file_name}_llama.txt"
        with open(file_path, 'w') as f:
            accuracy, precision, recall, f1 = calculate_metrics(merged_df_hotpot_llama)
            f.write(json.dumps({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }) + "\n")
        

        file_path = f"results/{file_name}_mistral.txt"
        with open(file_path, 'w') as f:
            accuracy, precision, recall, f1 = calculate_metrics(merged_df_hotpot_mistral)
            f.write(json.dumps({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }) + "\n")