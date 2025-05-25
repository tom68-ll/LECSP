import json
import torch
import os
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
import random
import argparse


model_id = "../model/models--mistralai--Mixtral-8x7B-Instruct-v0.1/snapshots/125c431e2ff41a156b9f9076f744d2f35dd6e67a"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id, load_in_8bit=True) # load_in_8bit=True
print("Model is on device:", model.device)

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', default=0,type=int)
parser.add_argument('--batch_size', default=16,type=int)
parser.add_argument('--input_max_length', default=4096,type=int)
args = parser.parse_args()

task_id = args.task_id

original_root_path = "../data/spider_task_stream_new"
output_root_path = "../data/intra_data/step1_out"

original_data_path = os.path.join(original_root_path,"task_"+str(task_id),"train.json")
output_path = os.path.join(output_root_path,"task_"+str(task_id)+".json")


batch_size = args.batch_size

def paraphrase_nlq(all_datas):
    prompt_all = []
    batch_result = []
    for data in all_datas:
        prompt_all.append("[INST]" + data["question"]+'\n'+'--Rephrase the question above in such a way that its meaning is preserved. Be mindful not to just replace words; offer a meaningful rewrite.'\
                        + "\n" + """--You will only give the rephrased question and do not generate extra information""" + "[/INST]\n")
    batches = [prompt_all[i:i + batch_size] for i in range(0, len(prompt_all), batch_size)]

    max_length = 0
    # make prompt
    for batch in tqdm(batches, total=len(batches)):
        
        inputs = tokenizer(batch, 
                           return_tensors="pt", 
                           padding=True,
                           truncation=True,
                           max_length=args.input_max_length)

        '''
        if inputs['input_ids'].size()[1] > max_length:
            max_length = inputs['input_ids'].size()[1]
            
            
    print("max length: {}".format(max_length))
    '''
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        batch_predictions = model.generate(
            **inputs,
            max_new_tokens=2048
        )

        for prediction in batch_predictions:
            output = tokenizer.decode(prediction, skip_special_tokens=True)
            batch_result.append(output)
        
    return batch_result

def caneted(unprocessed_nlq,unpacked_sql):
    splitted_by_double_newline = unprocessed_nlq.split('\n\n')

    # split
    final_split = [part.split('\n') for part in splitted_by_double_newline]

    # get all the split result
    final_sentences = [sentence for part in final_split for sentence in part]
    nlq_out = final_sentences[2]
    sql_out = unpacked_sql[1]
    return nlq_out,sql_out


if __name__ == '__main__':
    current_time = datetime.now()
    print("Current time:", current_time)

    with open(original_data_path,'r') as f :
        datas = json.load(f)
    all_datas = []
    for data in datas:
        if data["augment_data"]:
            for item in data["augment_data"]:
                all_datas.append({
                    "db_id": data["db_id"],
                    "question": item[0],
                    "query": item[1],
                })
    
    data_out = paraphrase_nlq(all_datas)
    
    assert len(all_datas) == len(data_out)
    output_data = []
    for data_i,data_o in zip(all_datas,data_out):
        output_data.append("####" + data_i["db_id"] + "####" + data_i["query"] + "####" + data_o)
    with open(output_path,'a') as f:
        json.dump(output_data,f)


file_path = os.path.join(dataset_path,bd_id,bd_id+'.sqlite')
for value_key in values_dict.keys():
    colomns_now = find_value_in_specific_tables(file_path, values_dict[value_key], tables_in_query)
    columns_in_values[values_dict[value_key]] = colomns_now
output_data = exchange_value_from_colomns(columns_in_values,values_dict,question_template,query_only_valued,file_path,former_values_dict)