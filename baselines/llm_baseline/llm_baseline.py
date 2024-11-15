import json
import re

import ollama

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

def clean_response(response):
    '''cleans the response if it contains a zero or 1 but skips if those characters appear in numbers such as years.
     this is based on typical cases we observed in responses and can be expanded based on the responses we see'''
    response= str(response).strip().lower()
    # pattern for zer0
    if re.search( r"(?<!\d)[\s\*]*0[\s\*]*(?!\d)", response):
        return 0
    if ( 'i cannot accurately answer' in response) or ('i cannot provide information' in response):
        return 0
    if re.search( r"(?<!\d)[\s\*]*1[\s\*]*(?!\d)", response):
        return 1
    if ( 'i can accurately answer' in response) or ('i can provide information' in response) or ('i can  answer' in response):
        return 1

    return 0 



qa_system_prompt= '''You are a question-answering prediction system. Your job is to predict whether you can accurately answer the question or not.  For each prompt, reply "1" if you can answer it and "0" if you cannot. Respond only with "0" or "1"''' 
# this function gets a response back from LLM, set the model to llama3 or mistral-nemo to get the response
def llm_predictor(system_prompt, user_prompt):
    response = ollama.generate(model='llama3',  system=  system_prompt,  prompt= user_prompt)                       
    return response['response']

# the prediction of llm models (mistral-nemo and llama3-8B) are gathereded for each prompt variation in the test set. 
# for hotpotqa use the PATH= 'prompt_set/hotpotqa/hotpotqa_llama3_8B_dataset_test.jsonl'
PATH= 'prompt_set/triviaqa/triviaqa_llama3_8B_dataset_test.jsonl'
data= read_jsonl(PATH)
print( 'Total number of prompts ', len(data))
# choose the name of the output file
response_alt= open('llama3_baseline_test_triviaqa.jsonl', 'a')
for ind, prompt in enumerate(data):
        if ind% 100==0:
            print('number of seen prompts', ind)
        response=llm_predictor( system_prompt=qa_system_prompt, user_prompt=prompt['prompt'] )
        response = clean_response(response)
        json_line={'prompt_id':prompt['prompt_id'], 'model':'llama3_8B', 'llm_baseline':response}
        write_jsonl_line(response_alt, json_line)

response_alt.close()