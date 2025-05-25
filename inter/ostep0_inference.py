from vllm import LLM, SamplingParams
import os
import json
import collections
import torch
import time
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import random
from prompt_text_to_sql.print_prompt import prompt_generate,multi_table_prompt_make
import re
import random

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

def calculate_union(now_idx):
    centers = []
    for idx in range(now_idx):
        path_of_this = os.path.join(data_cluster_path, "task_" + str(idx) + '.json')
        datai = load_json(path_of_this)
        for data in datai:
            center_tmp = []
            for item in data["center"]:
                if item != '{OP}':
                    center_tmp.append(item)
                elif '{OP}' not in center_tmp:
                    center_tmp.append(item)
                else:
                    continue
            centers.append(tuple(center_tmp))  # Use tuple instead of frozenset
    return set(centers)

def calculate_now(now_idx):
    centers = []
    path_of_this = os.path.join(data_cluster_path, "task_" + str(now_idx) + '.json')
    datai = load_json(path_of_this)
    for data in datai:
        center_tmp = []
        for item in data["center"]:
            if item != '{OP}':
                center_tmp.append(item)
            elif '{OP}' not in center_tmp:
                center_tmp.append(item)
            else:
                continue
        centers.append(tuple(center_tmp))  # Use tuple instead of frozenset
    return set(centers)

def calculate_diff(set1, set2):
    return set1 - set2


if __name__ == '__main__':
    
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens = 2048)
    
    
    #model_path =  "path_to_Mixtral-8x7B-Instruct-v0.1"
    #model_path = "path_to_Meta-Llama-3-8B-Instruct"
    model_path = "path_to_Mistral-7B-Instruct-v0.3"

    if "Mixtral" in model_path:
        start_token = "[INST]"
        end_token = "[/INST]"
    elif "Llama" in model_path:
        start_token = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
        end_token = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    elif "Mistral" in model_path:
        start_token = "[INST]"
        end_token = "[/INST]"
    
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
    #llm = LLM(model=model_path, tensor_parallel_size=4, gpu_memory_utilization=0.8)  # Adjusted utilization and tensor_parallel_size
    parser = argparse.ArgumentParser()

    parser.add_argument('--start_task_id', type=int,default=1)
    parser.add_argument('--end_task_id', type=int,default=1)
    parser.add_argument('--batch_size', type=int,default=16)
    parser.add_argument('--inference_type', type=str,default="V1",choices=["ordinary","no_OP","V1"])
    parser.add_argument('--input_max_length', type=int,default=4096)
    parser.add_argument('--order', type=int,default=0,choices=[0,1])

    args = parser.parse_args()

    root_data_pth = "../data/combine1_perm_1"
    data_cluster_path = "../data/cluster_result/cluster"
    wikisql_multi_root_path = '../data/wikisql/tables_made'
    root_save_path = "../data/only_task0_order0/Mistral/step0_out"
    name_orders = ['spider','wikisql','spider','wikisql','spider','wikisql','spider']
    #name_orders = ['wikisql','spider','spider','spider','spider','wikisql','wikisql']

    batch_size = args.batch_size
    start_task_id = args.start_task_id
    end_task_id = args.end_task_id
    task_num = end_task_id - start_task_id + 1

    remember = []
    for task_id in range(start_task_id, end_task_id+1):
        name_order = name_orders[task_id]
        print('\n','*'*20, f"Task_{task_id}", '*'*20)
        # read data
        task_test_pth = os.path.join(root_data_pth, f"task_{task_id}", "train.json")
        all_tests = []

        if task_id != 0:
            pattern_before = calculate_union(task_id)
            pattern_now = calculate_now(task_id)
            pattern_to_use = calculate_diff(pattern_before,pattern_now)
        else:
            pattern_now = calculate_now(task_id)
            pattern_to_use = pattern_now
        #pattern_to_use = pattern_now
        
        db_id_set = set()
        if name_order == 'spider':
            with open(task_test_pth, 'r') as f1:
                train_data = json.load(f1)
                for dt in train_data:
                    db_id_set.add(dt['example']["db_id"])
                    db_name = dt['example']["db_id"]
                    gold_sql = dt['example']["query"]
                    question = dt['example']["question"].capitalize()
                    all_tests.append({"db_name": db_name, "gold_sql": gold_sql, "question": question})
        elif name_order == 'wikisql':
            text_set = list()
            with open(task_test_pth, 'r') as f1:
                train_data = json.load(f1)
                for dt in train_data:
                    if dt['text'].split('|')[0] not in text_set:
                        db_id_set.add(dt['example']["db_id"])
                        text_set.append(dt['text'].split('|')[0])
                        db_name = dt['example']["db_id"]
                        gold_sql = dt['example']["query"]
                        question = dt['example']["question"].capitalize()
                        all_tests.append({"db_name": db_name, "gold_sql": gold_sql, "question": question})
            # for line in f1:
            #     smp = json.loads(line)
            #     all_tests.append(smp)
        # print(f"Test_num: {len(all_tests)}")

        prompt_all = []
        ''''
        db_id_list_tmp = list(db_id_set)
        if len(db_id_list_tmp) > 70:
            db_id_list = random.sample(db_id_list_tmp, 70)
        else:
            db_id_list = random.sample(db_id_list_tmp, int(len(db_id_list_tmp) * 0.7))
        '''
        db_id_list = list(db_id_set)
        #db_id_list = random.sample(db_id_list,15)
        # make prompt
        label_list = []
        if name_order == 'spider':
            #dAS1
            for db_id_now in tqdm(db_id_list):
                for pattern in tqdm(pattern_to_use):
                    prompt = prompt_generate(db_id_now,pattern,prompt_db="CreateTableInsertRow",prompt_type=args.inference_type).replace("\"","").replace("\'","").replace("`","")
                    prompt_all.append(prompt)
                    label_list.append("dAS1")
        elif name_order == 'wikisql':
            db_id_list_forwiki = []
            wikisql_multi_path = os.path.join(wikisql_multi_root_path,f'task_{task_id}_db')
            multi_table_pattern = []
            single_table_pattern = []
            for pattern in pattern_to_use:
                if 'JOIN' in pattern:
                    multi_table_pattern.append(pattern)
                else:
                    single_table_pattern.append(pattern)
            #dAS1
            for db_id_now in tqdm(db_id_list):
                for pattern in tqdm(single_table_pattern):
                    prompt = prompt_generate(db_id_now,pattern,prompt_db="CreateTableInsertRow",prompt_type=args.inference_type).replace("\"","").replace("\'","").replace("`","")
                    prompt_all.append(prompt)
                    label_list.append("dAS1")
                    db_id_list_forwiki.append(db_id_now)
            #dAS2
            
            for pattern in tqdm(multi_table_pattern):
                dataset_names = os.listdir(wikisql_multi_path)
                for dataset_name in dataset_names:
                    print(dataset_name)
                    db_id_list_forwiki.append(dataset_name.split('.')[0])
                    multi_input_path = os.path.join(wikisql_multi_path,dataset_name)
                    label_list.append("dAS2")
                    prompt_all.append(multi_table_prompt_make(multi_input_path,pattern,prompt_db="CreateTableInsertRow",prompt_type=args.inference_type).replace("\"","").replace("\'","").replace("`",""))
            
        print("len(db_id_set): {}".format(len(db_id_set)))
        print("len(pattern_to_use): {}".format(len(pattern_to_use)))
        print("len(prompt_all): {}".format(len(prompt_all)))
        

        prompt_path = "./prompt/all_trains_of" + str(task_id) + '.json'
        with open(prompt_path,'w') as f:
            json.dump(prompt_all,f)
        
        schema_path = "./info_for_step1_former/label_of" + str(task_id) + '.json'
        with open(schema_path,'w') as f:
            json.dump(label_list,f)
        
        max_length = 0
        for item in prompt_all:
            if len(re.split(" |\n",item)) > max_length:
                max_length = len(re.split(" |\n",item))
        print(f"max_length:{max_length}")
        
        
        
        batches = [prompt_all[i:i + batch_size] for i in range(0, len(prompt_all), batch_size)]

        result = []
        save_path = os.path.join(root_save_path, f"task_{task_id}.txt")

        max_len = 0
        lengths = 0
        number = 0
        batch_result = []
        db_id_idx = 0
        if 'Mistral' not in model_path:
            for batch in tqdm(batches, total=len(batches)):
                print("Loading...")
                os.system("nvidia-smi")
                outputs = llm.generate(batch, sampling_params)

                output_start = "***********************************************\n"
                # Print the outputs.
                for output in outputs:
                    prompt = output.prompt
                    generated_text = output.outputs[0].text
                    if name_order == 'spider':
                        result.append("db_id:" + str(db_id_list[db_id_idx // len(pattern_to_use)]) + "\n" + generated_text)
                        with open(save_path, "a", encoding="utf-8") as file:
                            file.write(output_start + "db_id:" + str(db_id_list[db_id_idx // len(pattern_to_use)]) + "\n" + prompt + "\n" + "generated_text_start:\n" + generated_text + "\n")
                        db_id_idx += 1
                    elif name_order == 'wikisql':
                        result.append("db_id:" + str(db_id_list_forwiki[db_id_idx]) + "\n" + generated_text)
                        with open(save_path, "a", encoding="utf-8") as file:
                            file.write(output_start + "db_id:" + str(db_id_list_forwiki[db_id_idx]) + "\n" + prompt + "\n" + "generated_text_start:\n" + generated_text + "\n")
                        db_id_idx += 1
            
        else:
            for batch in tqdm(batches, total=len(batches)):
                print("Loading...")
                os.system("nvidia-smi")
                inputs = tokenizer(batch, 
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
                for output in batch_predictions:
                    output_decoded = tokenizer.decode(output, skip_special_tokens=True)
                    print(output_decoded)
                    prompt = output_decoded.split("If {OP} is in the keyword, it means that the generated SQL should include operators, such as >, < and = ")[0] + "If {OP} is in the keyword, it means that the generated SQL should include operators, such as >, < and = " + '\n'
                    generated_text = output_decoded.split("If {OP} is in the keyword, it means that the generated SQL should include operators, such as >, < and = ")[1].strip()
                    if name_order == 'spider':
                        result.append("db_id:" + str(db_id_list[db_id_idx // len(pattern_to_use)]) + "\n" + generated_text)
                        with open(save_path, "a", encoding="utf-8") as file:
                            file.write(output_start + "db_id:" + str(db_id_list[db_id_idx // len(pattern_to_use)]) + "\n" + prompt + "\n" + "generated_text_start:\n" + generated_text + "\n")
                        db_id_idx += 1
                    elif name_order == 'wikisql':
                        result.append("db_id:" + str(db_id_list_forwiki[db_id_idx]) + "\n" + generated_text)
                        with open(save_path, "a", encoding="utf-8") as file:
                            file.write(output_start + "db_id:" + str(db_id_list_forwiki[db_id_idx]) + "\n" + prompt + "\n" + "generated_text_start:\n" + generated_text + "\n")
                        db_id_idx += 1

    









