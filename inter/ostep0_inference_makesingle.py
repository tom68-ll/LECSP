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
import sqlite3

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

class get_text:
    def __init__(self,db_path,question,conn=None):
        self.db_path = db_path
        self.question = question
        self.conn = conn

    def get_db_schema(self):
        """
        Extracts the database schema (tables and columns) from a SQLite database.

        Returns:
        dict: A dictionary with table names as keys and lists of column names as values.
        """
        schema = {}
        if not self.conn:
            # Connect to the SQLite database
            conn = sqlite3.connect(self.db_path)
        else:
            conn = self.conn
        cursor = conn.cursor()
        
        # Query for all tables in the database
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        # Loop through tables and get column details
        for (table_name,) in tables:
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            schema[table_name] = ', '.join(column[1] for column in columns)
        
        cursor.close()
        if not self.conn:
            conn.close()
        return schema

    def generate_output(self):
        """
        Generates formatted text combining database schema and a specific question.

        Parameters:
        schema (dict): Database schema as a dictionary.
        question (str): A specific question related to the database.

        Returns:
        str: Formatted text combining the schema and the question.
        """
        # Format the schema into text
        schema_text = "; ".join([f"{table}: {columns}" for table, columns in self.schema.items()])
        
        # Combine the schema text and the question to form the output
        output_text = f"{schema_text} | {self.question}"
        return output_text

    def get_text(self):
        self.schema = self.get_db_schema()
        output = self.generate_output()
        return output

if __name__ == '__main__':
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95,max_tokens = 500)
    
    
    #model_path =  "path_to_Mixtral-8x7B-Instruct-v0.1"
    model_path = "path_to_Meta-Llama-3-8B-Instruct"
    #model_path = "path_to_Qwen2-72B-Instruct"

    if "Mixtral" in model_path:
        start_token = "[INST]"
        end_token = "[/INST]"
    elif "Llama" in model_path:
        start_token = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
        end_token = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    elif "Qwen" in model_path:
        start_token = "||||"
        end_token = "||||"
    
    llm = LLM(model=model_path, tensor_parallel_size=1)
    #llm = LLM(model=model_path, tensor_parallel_size=4, gpu_memory_utilization=0.8)  # Adjusted utilization and tensor_parallel_size
    parser = argparse.ArgumentParser()

    parser.add_argument('--start_task_id', type=int,default=0)
    parser.add_argument('--end_task_id', type=int,default=6)
    parser.add_argument('--batch_size', type=int,default=16)
    parser.add_argument('--inference_type', type=str,default="V1",choices=["ordinary","no_OP","V1"])
    parser.add_argument('--input_max_length', type=int,default=4096)
    parser.add_argument('--order', type=int,default=0,choices=[0,1])

    args = parser.parse_args()

    root_data_pth = "../data/combine1_perm_1"
    data_cluster_path = "../data/cluster_result/cluster"
    wikisql_multi_root_path = '../data/wikisql/tables_made'
    root_save_path = "../data/only_task0_order0/llama/step0_out"
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
        '''
        if task_id != 0:
            pattern_before = calculate_union(task_id)
            pattern_now = calculate_now(task_id)
            pattern_to_use = calculate_diff(pattern_before,pattern_now)
        else:
            pattern_to_use = calculate_now(task_id)
        '''
        pattern_before = calculate_union(task_id)
        #A
        pattern_now = calculate_now(task_id)
        #dA
        pattern_to_use = calculate_diff(pattern_before,pattern_now)
        
        
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
        db_id_set = list(db_id_set)
        #db_id_list = random.sample(db_id_list,15)
        # make prompt
        label_list = []
        if name_order == 'spider':

            #prompt pattern to use
            prompt_db_pairs = {}
            for db_id_now in tqdm(db_id_set):
                #!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                if task_id == 0:
                    for pattern in tqdm(pattern_now):
                        prompt = prompt_generate(db_id_now,pattern,prompt_db="CreateTableInsertRow",prompt_type=args.inference_type).replace("\"","").replace("\'","").replace("`","")
                        prompt_all.append(prompt)
                        if prompt.split('--Create 10 suitable SQL queries that simultaneously include the SQL keywords:')[0] not in prompt_db_pairs.keys():
                            prompt_db_pairs[prompt.split('--Create 10 suitable SQL queries that simultaneously include the SQL keywords:')[0]] = db_id_now
                else:
                    for pattern in tqdm(pattern_to_use):
                        prompt = prompt_generate(db_id_now,pattern,prompt_db="CreateTableInsertRow",prompt_type=args.inference_type).replace("\"","").replace("\'","").replace("`","")
                        prompt_all.append(prompt)
                        if prompt.split('--Create 10 suitable SQL queries that simultaneously include the SQL keywords:')[0] not in prompt_db_pairs.keys():
                            prompt_db_pairs[prompt.split('--Create 10 suitable SQL queries that simultaneously include the SQL keywords:')[0]] = db_id_now
            #select prompt
            pattern_not_join = []
            for item in pattern_to_use:
                if 'JOIN' not in item:
                    pattern_not_join.append(item)
            prompt_set = []
            prompt_setlist = []
            for prompt in prompt_all:
                if prompt.split('\n\n')[0] not in prompt_set:
                    prompt_set.append(prompt.split('\n\n')[0])
                    prompt_setlist.append(prompt)
            prompt_set = prompt_set[:int(len(prompt_set)/3)]
            prompt_setlist = prompt_setlist[:int(len(prompt_set))]
            prompt_single = []
            db_id_list = []
            schema_list = []
            
            for prompt in prompt_setlist:
                prompt_after = '. And provide a corresponding and accurate natural' + prompt.split('--Create 10 suitable SQL queries that simultaneously include the SQL keywords: ')[1].split('. And provide a corresponding and accurate natural')[1]
                prompt_tmp = prompt.split('--Create 10 suitable SQL queries that simultaneously include the SQL keywords: ')[0].strip(start_token)
                tables = prompt_tmp.split('create table')[1:]
                table_chosen = random.sample(tables,2)
                tables_tmp = []
                for table in table_chosen:
                    table_rows = table.split('\n')
                    table_tmp = []
                    for row in table_rows:
                        if 'foreign key' in row:
                            continue
                        table_tmp.append(row)
                    tables_tmp.append('\n'.join(table_tmp))
                table_chosen = tables_tmp

                
                db_id_tmp = prompt_db_pairs[prompt.split('--Create 10 suitable SQL queries that simultaneously include the SQL keywords:')[0]]
                db_path = os.path.join('../data/spider/database',db_id_tmp,db_id_tmp+'.sqlite')
                get_text1 = get_text(db_path=db_path,question='')
                text = get_text1.get_text().strip(' | ')
                schema_list_tmp = []
                for table in table_chosen:
                    table_name = table.strip().split('(')[0].strip()
                    for schema_tmp in text.split(';'):
                        if table_name.lower() == schema_tmp.split(':')[0].strip().lower():
                            schema_list_tmp.append(schema_tmp.strip())
                            break
                if len(schema_list_tmp) < 2:
                    print("schema_list_not enough!\n")
                    print(prompt)
                    print(tables)
                    print(table_chosen)
                    print(text,table_chosen)
                    print(schema_list_tmp)
                for pattern in pattern_not_join:
                    db_id_list.append(prompt_db_pairs[prompt.split('--Create 10 suitable SQL queries that simultaneously include the SQL keywords:')[0]])
                    db_id_list.append(prompt_db_pairs[prompt.split('--Create 10 suitable SQL queries that simultaneously include the SQL keywords:')[0]])
                    prompt_single.append(f"{start_token}create table{table_chosen[0]}--Create 10 suitable SQL queries that simultaneously include the SQL keywords: {' '.join(pattern)}{prompt_after}")
                    prompt_single.append(f"{start_token}create table{table_chosen[1]}--Create 10 suitable SQL queries that simultaneously include the SQL keywords: {' '.join(pattern)}{prompt_after}")
                    schema_list.append(schema_list_tmp[0])
                    schema_list.append(schema_list_tmp[1])
                    label_list.append("dAS2")
                    label_list.append("dAS2")


            #prompt pattern before
            prompt_db_pairs = {}
            for db_id_now in tqdm(db_id_set):
                for pattern in tqdm(pattern_now):
                    prompt = prompt_generate(db_id_now,pattern,prompt_db="CreateTableInsertRow",prompt_type=args.inference_type).replace("\"","").replace("\'","").replace("`","")
                    prompt_all.append(prompt)
                    if prompt.split('--Create 10 suitable SQL queries that simultaneously include the SQL keywords:')[0] not in prompt_db_pairs.keys():
                        prompt_db_pairs[prompt.split('--Create 10 suitable SQL queries that simultaneously include the SQL keywords:')[0]] = db_id_now
            #select prompt
            pattern_not_join = []
            for item in pattern_to_use:
                if 'JOIN' not in item:
                    pattern_not_join.append(item)
            prompt_set = []
            prompt_setlist = []
            for prompt in prompt_all:
                if prompt.split('\n\n')[0] not in prompt_set:
                    prompt_set.append(prompt.split('\n\n')[0])
                    prompt_setlist.append(prompt)
            prompt_set = prompt_set[:int(len(prompt_set)/3)]
            prompt_setlist = prompt_setlist[:int(len(prompt_set))]
            for prompt in prompt_setlist:
                prompt_after = '. And provide a corresponding and accurate natural' + prompt.split('--Create 10 suitable SQL queries that simultaneously include the SQL keywords: ')[1].split('. And provide a corresponding and accurate natural')[1]
                prompt_tmp = prompt.split('--Create 10 suitable SQL queries that simultaneously include the SQL keywords: ')[0].strip(start_token)
                tables = prompt_tmp.split('create table')[1:]
                table_chosen = random.sample(tables,2)
                tables_tmp = []
                for table in table_chosen:
                    table_rows = table.split('\n')
                    table_tmp = []
                    for row in table_rows:
                        if 'foreign key' in row:
                            continue
                        table_tmp.append(row)
                    tables_tmp.append('\n'.join(table_tmp))
                table_chosen = tables_tmp

                
                db_id_tmp = prompt_db_pairs[prompt.split('--Create 10 suitable SQL queries that simultaneously include the SQL keywords:')[0]]
                db_path = os.path.join('../data/spider/database',db_id_tmp,db_id_tmp+'.sqlite')
                get_text1 = get_text(db_path=db_path,question='')
                text = get_text1.get_text().strip(' | ')
                schema_list_tmp = []
                for table in table_chosen:
                    table_name = table.strip().split('(')[0].strip()
                    for schema_tmp in text.split(';'):
                        if table_name.lower() == schema_tmp.split(':')[0].strip().lower():
                            schema_list_tmp.append(schema_tmp.strip())
                            break
                if len(schema_list_tmp) < 2:
                    for pattern in pattern_not_join:
                        db_id_list.append(prompt_db_pairs[prompt.split('--Create 10 suitable SQL queries that simultaneously include the SQL keywords:')[0]])
                        prompt_single.append(f"{start_token}create table{table_chosen[0]}--Create 10 suitable SQL queries that simultaneously include the SQL keywords: {' '.join(pattern)}{prompt_after}")
                        schema_list.append(schema_list_tmp[0])
                        label_list.append("AS2")
                else:
                    for pattern in pattern_not_join:
                        db_id_list.append(prompt_db_pairs[prompt.split('--Create 10 suitable SQL queries that simultaneously include the SQL keywords:')[0]])
                        db_id_list.append(prompt_db_pairs[prompt.split('--Create 10 suitable SQL queries that simultaneously include the SQL keywords:')[0]])
                        prompt_single.append(f"{start_token}create table{table_chosen[0]}--Create 10 suitable SQL queries that simultaneously include the SQL keywords: {' '.join(pattern)}{prompt_after}")
                        prompt_single.append(f"{start_token}create table{table_chosen[1]}--Create 10 suitable SQL queries that simultaneously include the SQL keywords: {' '.join(pattern)}{prompt_after}")
                        schema_list.append(schema_list_tmp[0])
                        schema_list.append(schema_list_tmp[1])
                        label_list.append("AS2")
                        label_list.append("AS2")
            prompt_all = prompt_single


        elif name_order == 'wikisql':
            schema_list = []
            db_id_list = []
            wikisql_multi_path = os.path.join(wikisql_multi_root_path,f'task_{task_id}_db')
            multi_table_pattern = []
            single_table_pattern = []
            prompt_db_pairs = {}
            pattern_before = calculate_union(task_id)
            pattern_now = calculate_now(task_id)
            pattern_to_use = calculate_diff(pattern_before,pattern_now)
            for pattern in pattern_to_use:
                if 'JOIN' in pattern:
                    multi_table_pattern.append(pattern)
                else:
                    single_table_pattern.append(pattern)
            '''
            for db_id_now in tqdm(db_id_list):
                for pattern in tqdm(single_table_pattern):
                    prompt = prompt_generate(db_id_now,pattern,prompt_db="CreateTableInsertRow",prompt_type=args.inference_type).replace("\"","").replace("\'","").replace("`","")
                    prompt_all.append(prompt)
                    db_id_list_forwiki.append(db_id_now)
            '''
            dataset_names = []
            #dAS2
            for pattern in tqdm(single_table_pattern):
                dataset_names = os.listdir(wikisql_multi_path)
                for dataset_name in dataset_names:
                    db_id_list.append(dataset_name.split('.')[0])
                    multi_input_path = os.path.join(wikisql_multi_path,dataset_name)
                    prompt_all.append(multi_table_prompt_make(multi_input_path,pattern,prompt_db="CreateTableInsertRow",prompt_type=args.inference_type).replace("\"","").replace("\'","").replace("`",""))
                    label_list.append("dAS2")
            prompt_setlist = dataset_names
            pattern_to_use = calculate_now(task_id)
            #AS2
            for pattern in tqdm(pattern_to_use):
                dataset_names = os.listdir(wikisql_multi_path)
                for dataset_name in dataset_names:
                    db_id_list.append(dataset_name.split('.')[0])
                    multi_input_path = os.path.join(wikisql_multi_path,dataset_name)
                    prompt_all.append(multi_table_prompt_make(multi_input_path,pattern,prompt_db="CreateTableInsertRow",prompt_type=args.inference_type).replace("\"","").replace("\'","").replace("`",""))
                    label_list.append("AS2")
            '''
            for pattern in tqdm(multi_table_pattern):
                dataset_names = os.listdir(wikisql_multi_path)
                for dataset_name in dataset_names:
                    db_id_list_forwiki.append(dataset_name.split('.')[0])
                    multi_input_path = os.path.join(wikisql_multi_path,dataset_name)
                    prompt = multi_table_prompt_make(multi_input_path,pattern,prompt_db="CreateTableInsertRow",prompt_type=args.inference_type).replace("\"","").replace("\'","").replace("`","")
                    prompt_all.append(prompt)
                    if prompt.split('--Create 10 suitable SQL queries that simultaneously include the SQL keywords:')[0] not in prompt_db_pairs.keys():
                        prompt_db_pairs[prompt.split('--Create 10 suitable SQL queries that simultaneously include the SQL keywords:')[0]] = dataset_name.split('.')[0]
            pattern_to_use = single_table_pattern
            prompt_set = []
            prompt_setlist = []
            for prompt in prompt_all:
                if prompt.split('\n\n')[0] not in prompt_set:
                    prompt_set.append(prompt.split('\n\n')[0])
                    prompt_setlist.append(prompt)
            prompt_set = prompt_set[:int(len(prompt_set)/3)]
            prompt_setlist = prompt_setlist[:int(len(prompt_set))]
            prompt_single = []
            db_id_list = []
            for prompt in prompt_setlist:
                prompt_after = '. And provide a corresponding and accurate natural' + prompt.split('--Create 10 suitable SQL queries that simultaneously include the SQL keywords: ')[1].split('. And provide a corresponding and accurate natural')[1]
                prompt_tmp = prompt.split('--Create 10 suitable SQL queries that simultaneously include the SQL keywords: ')[0].strip('[INST]')
                tables = prompt_tmp.split('create table')[1:]
                table_chosen = random.sample(tables,2)
                tables_tmp = []
                for table in table_chosen:
                    table_rows = table.split('\n')
                    table_tmp = []
                    for row in table_rows:
                        if 'foreign key' in row:
                            continue
                        table_tmp.append(row)
                    tables_tmp.append('\n'.join(table_tmp))
                table_chosen = tables_tmp
                for pattern in single_table_pattern:
                    db_id_list.append(prompt_db_pairs[prompt.split('--Create 10 suitable SQL queries that simultaneously include the SQL keywords:')[0]])
                    db_id_list.append(prompt_db_pairs[prompt.split('--Create 10 suitable SQL queries that simultaneously include the SQL keywords:')[0]])
                    prompt_single.append(f"[INST]create table{table_chosen[0]}--Create 10 suitable SQL queries that simultaneously include the SQL keywords: {' '.join(pattern)}{prompt_after}")
                    prompt_single.append(f"[INST]create table{table_chosen[1]}--Create 10 suitable SQL queries that simultaneously include the SQL keywords: {' '.join(pattern)}{prompt_after}")
            prompt_all = prompt_single
            '''
        print("len(db_id_set): {}".format(len(prompt_setlist)))
        print("len(pattern_to_use): {}".format(len(pattern_to_use)))
        print("len(prompt_all): {}".format(len(prompt_all)))
        print("len(db_id_list): {}".format(len(db_id_list)))
        print(f"schema count:{len(schema_list)}")

        if "Qwen" in model_path:
            for idx, prompt in enumerate(prompt_all):
                prompt_all[idx] = re.sub(r'\|\|\|\|', '', prompt)

        
        #prompt_path = "all_trains_of" + str(task_id) + '.json'
        prompt_path = "./prompt_forsingle/all_trains_of" + str(task_id) + '.json'
        with open(prompt_path,'w') as f:
            json.dump(prompt_all,f)
        
        #schema_path = "all_schema_of" + str(task_id) + '.json'
        schema_path = "./prompt_forsingle/all_schema_of" + str(task_id) + '.json'
        with open(schema_path,'w') as f:
            json.dump(schema_list,f)

        #schema_path = "label_of" + str(task_id) + '.json'
        schema_path = "./info_for_step1/label_of" + str(task_id) + '.json'
        with open(schema_path,'w') as f:
            json.dump(label_list,f)
        print("label_list ok")
        print(f"len {len(label_list)}")
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
        for batch in tqdm(batches, total=len(batches)):
            print("Loading...")
            os.system("nvidia-smi")
            outputs = llm.generate(batch, sampling_params)

            output_start = "***********************************************\n"
            # Print the outputs.
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                result.append("db_id:" + str(db_id_list[db_id_idx]) + "\n" + generated_text)
                with open(save_path, "a", encoding="utf-8") as file:
                    file.write(output_start + "db_id:" + str(db_id_list[db_id_idx]) + "\n" + prompt + "\n" + "generated_text_start:\n" + generated_text + "\n")
                db_id_idx += 1
        

