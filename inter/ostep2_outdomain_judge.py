import os
import json
import collections
import torch
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

from step2_prompt.print_prompt import prompt_generate,prompt_generate_forwiki_multi
from vllm import LLM, SamplingParams

import os


def load_data(meta_data, batch_size):
    data = []
    cnt = 0
    current_batch = []
    for d in meta_data:
        current_batch.append(d)
        cnt += 1
        if cnt >= batch_size:
            data.append(current_batch)
            current_batch = []
            cnt = 0
    if len(current_batch) > 0:
        data.append(current_batch)
    return data



def load_json(json_file) -> list:
    with open(json_file, 'r', encoding='utf-8') as f:
        ex_list = json.load(f)
    return ex_list

def get_db_schema(bench_roots) -> dict:
    db_schema = collections.defaultdict(dict)
    for bench_root in bench_roots:
        tables_json = load_json(bench_root + '/tables.json')
        for table_json in tables_json:
            db_id = table_json['db_id']
            db_schema[db_id] = collections.defaultdict(dict)

            table_id_to_column_ids = collections.defaultdict(list)
            column_id_to_column_name = {}
            column_id_to_table_id = {}
            for column_id, table_column in enumerate(table_json['column_names_original']):
                table_id = table_column[0]
                column_name = table_column[1]
                column_id_to_column_name[column_id] = column_name.replace(" ", "_")
                table_id_to_column_ids[table_id].append(column_id)
                column_id_to_table_id[column_id] = table_id

            column_id_to_column_type = {}
            if len(table_json['column_types']) < len(table_json['column_names_original']):
                table_json['column_types'] = ["text"] + table_json['column_types']
            for column_id, column_type in enumerate(table_json['column_types']):
                column_id_to_column_type[column_id] = column_type

            table_id_to_table_name = {}
            if len(table_json['table_names_original']) == 0:
                table_name_list = [db_id]
            else:
                table_name_list = table_json['table_names_original']
            for table_id, table_name in enumerate(table_name_list):
                table_id_to_table_name[table_id] = table_name.replace(" ", "_")

            primary_keys = table_json['primary_keys']
            foreign_keys = {}
            for column_id, referenced_column_id in table_json['foreign_keys']:
                foreign_keys[column_id] = referenced_column_id

            for table_id in table_id_to_table_name.keys():
                table_name = table_id_to_table_name[table_id]
                for column_id in table_id_to_column_ids[table_id]:
                    column_name = column_id_to_column_name[column_id]
                    column_info = {
                        'type': column_id_to_column_type[column_id],
                        'is_primary_key': column_id in primary_keys,
                        'is_foreign_key': column_id in foreign_keys.keys(),
                    }
                    if column_info['is_foreign_key']:
                        referenced_table = table_id_to_table_name[column_id_to_table_id[foreign_keys[column_id]]]
                        referenced_column = column_id_to_column_name[foreign_keys[column_id]]
                        column_info['reference'] = '.'.join([referenced_table, referenced_column])
                    db_schema[db_id][table_name][column_name] = column_info
    return db_schema

def contains_letter(s):
    return any(char.isalpha() for char in s)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int,default=0)
    parser.add_argument('--batch_size', type=int,default=32)
    parser.add_argument('--input_max_length', type=int,default=1200)
    parser.add_argument('--order', type=int,default=0,choices=[0,1])
    args = parser.parse_args()

    table_data_pth = "../data/spider"

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens = 500)
    #model_path =  "../model/Mixtral-8x7B-Instruct-v0.1"
    #model_path = "../model/Meta-Llama-3-8B-Instruct"
    model_path = "../model/Mistral-7B-Instruct-v0.3"
    if "Mistral" not in model_path:
        llm = LLM(model=model_path, tensor_parallel_size=1)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        #model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token


    if args.order == 0:
        root_save_path = "../data/only_task0_order0/Mistral/step2_out"
        root_data_pth = "../data/only_task0_order0/Mistral/step1_out"
    else:
        root_save_path = "../data/inter_data" + str(args.order) + "/step2_out"
        root_data_pth = "../data/inter_data" + str(args.order) + "/step1_out"

    name_orders = ['spider','wikisql','spider','wikisql','spider','wikisql','spider']
    #name_orders = ['wikisql','spider','spider','spider','spider','wikisql','wikisql']
    #generate prompts
    batch_size = args.batch_size
    #task_id = args.task_id

    for task_id in range(7):
        name_order = name_orders[task_id]

        remember = []

        print('\n','*'*20, f"Task_{task_id}", '*'*20)
        task_test_pth = os.path.join(root_data_pth, "task_" + str(task_id) + ".json")
        all_tests = []
        #db_schema = get_db_schema([table_data_pth])

        with open(task_test_pth, 'r') as f1:
            test_data = json.load(f1)
            for dt in test_data:
                db_name = dt["db_id"]
                gold_sql = dt["query"]
                question = dt["question"].capitalize()
                all_tests.append({"db_name": db_name, "gold_sql": gold_sql, "question": question,"text":dt["text"],"label":dt["label"]})

        # make prompt
        if name_order =='spider':
            for item in tqdm(all_tests):
                item["prompt"] = prompt_generate(item["db_name"],item["question"],item["gold_sql"],prompt_db="CreateTableInsertRow")
        elif name_order == 'wikisql':
            for item in tqdm(all_tests):
                if contains_letter(item["db_name"]):
                    item["prompt"] = prompt_generate_forwiki_multi(item["db_name"],item["question"],item["gold_sql"],prompt_db="CreateTableInsertRow",task_id=task_id)
                else:
                    item["prompt"] = prompt_generate(item["db_name"],item["question"],item["gold_sql"],prompt_db="CreateTableInsertRow")
        with open("all_tests.json",'w') as f:
            json.dump(all_tests,f)

        batches = [all_tests[i:i + batch_size] for i in range(0, len(all_tests), batch_size)]

        result = []
        save_path = os.path.join(root_save_path, f"task_{task_id}.json")

        data_idx = 0
        if "Mistral" not in model_path:
            for batch in tqdm(batches, total=len(batches)):
                print("Loading...")
                os.system("nvidia-smi")
                prompts = [data['prompt'] for data in batch]
                outputs = llm.generate(prompts, sampling_params)

                output_start = "***********************************************\n"
                # Print the outputs.
                for idx,output in enumerate(outputs):
                    prompt = output.prompt
                    generated_text = output.outputs[0].text
                    batch[idx]['result'] = generated_text
        else:
            for batch in tqdm(batches, total=len(batches)):
                print("Loading...")
                os.system("nvidia-smi")
                prompts = [data['prompt'] for data in batch]
                inputs = tokenizer(prompts, 
                                return_tensors="pt", 
                                padding=True,
                                truncation=True,
                                max_length=2048)
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                batch_predictions = model.generate(
                    **inputs,
                    max_new_tokens=2048
                )

                output_start = "***********************************************\n"
                # Print the outputs.
                for idx,output in enumerate(batch_predictions):
                    output_decoded = tokenizer.decode(output, skip_special_tokens=True)
                    generated_text = output_decoded.split("If no, please provide the correct SQL statements.")[1].strip()
                    batch[idx]['result'] = generated_text
        
        output_to_write = [item for sublist in batches for item in sublist]

        with open(save_path, "w", encoding="utf-8") as file:
            json.dump(output_to_write,file)
        



