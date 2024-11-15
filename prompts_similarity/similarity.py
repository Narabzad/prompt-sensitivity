

import json
from sentence_transformers import SentenceTransformer

# utils functions

def write_jsonl_line(open_file_object, json_line):
        '''a helper function to write a json line in a file object '''
        json.dump(json_line, open_file_object)
        open_file_object.write('\n')

def read_jsonl(file_path):
    '''a helper function to read  data from a jsonl file '''
    data=[]
    with  open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data





# Run for both TriviaQA and HotpotQA dataset
dataset_address= 'prompt_set_triviaqa_llama3_8B_dataset_test.jsonl'
original_prompts=[]
prompts={} # gives access to the content of the prompts by prompt_id
for i in dataset_address:
    prompts[i['prompt_id']] = i['prompt']
    if 'alt' not in i['prompt_id']:
        original_prompts.append(i['prompt_id'])

# create a dictionary representation of the original_prompts and their related variations       
data= {key: [] for key in original_prompts}
for i in dataset_address:
    if 'alt' in i['prompt_id']:
        data[i['reference_prompt_id']].append(i['prompt_id']) 

def get_variations(prompt_id):
    '''for any prompt_id returns two lists: the first contains prompts.
      the first item is the prompt for the prompt_id and the rest are the prompts for its variations. 
      The second list returns the prompt_id and the prompt_ids for its variations'''
    variations_ids = data[prompt_id] 
    return  [prompts[prompt_id]]+[prompts[i] for i in variations_ids], [ prompt_id]+ variations_ids


similarity_file= open('similarities_test_triviaqa.jsonl', 'a')
# Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")
for index, prompt_id in enumerate(original_prompts):
    sentences,  variation_ids = get_variations(prompt_id)

    # Calculate embeddings 
    embeddings = model.encode(sentences)

    # Calculate the embedding similarities
    similarities = model.similarity(embeddings[0, :], embeddings)

    for ind, i in enumerate(similarities[0]):
        json_line= {'prompt_id': variation_ids[ind], 'reference_prompt_id':prompt_id,  'similarity': i.item(), 'model': 'mini'}
        write_jsonl_line(similarity_file,json_line )

similarity_file.close()
