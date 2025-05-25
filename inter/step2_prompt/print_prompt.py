import argparse
#from database_prompt_construction import generate_db_prompt
from step2_prompt.database_prompt_construction import generate_db_prompt,generate_prompt,get_foreign_key_info,generate_table_prompt
#from sql_generation import get_prompt_length
#from utils import spider_train_db_ids, spider_dev_db_ids
from step2_prompt.utils import spider_train_db_ids, spider_dev_db_ids
import os
import sqlite3

def prompt_generate(db_id,NLQ,SQL,prompt_db="CreateTableSelectCol"):
    prompt_db = prompt_db
    db_id = db_id
    if db_id in spider_dev_db_ids:
        dataset = "spider-dev"
    elif db_id in spider_train_db_ids:
        dataset = "spider-train"
    else:
        dataset = "wikisql"

    print(db_id)
    if prompt_db in ["Table(Columns)", "Columns=[]", "Columns=[]+FK", "CreateTable"]:
        limit_value = 0
    elif prompt_db in ["CreateTableInsertRow", "CreateTableSelectRow", "CreateTableSelectCol"]:
        limit_value = 0
    else:
        raise ValueError("Unknown prompt_db")
    prompt_length_by_db = {}
    prompt = generate_db_prompt(dataset, db_id, prompt_db=prompt_db, limit_value=limit_value, normalization=True)
    
    prompt = "[INST]" + prompt + "Natural Language Question: "+NLQ + '\n' + "SQL query: "+ SQL + "\n\n" + \
        "--Do the above natural language questions correspond to the SQL queries? Please answer only yes or no, without any explanation. If no, please provide the correct SQL statements." + "[/INST]"
    '''  
    prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>" + prompt + "Natural Language Question: "+NLQ + '\n' + "SQL query: "+ SQL + "\n\n" + \
        "--Do the above natural language questions correspond to the SQL queries? Please answer only yes or no, without any explanation. If no, please provide the correct SQL statements." + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    '''
    '''
    prompt_len = get_prompt_length(prompt)
    print("prompt length:", prompt_len)
    '''
    return prompt


def extract_create_table_prompt_forwiki(prompt_db, db_id, cursor, limit_value=3, normalization=True):
    
    # get all the table info
    table_query = "SELECT * FROM sqlite_master WHERE type='table';"
    tables = cursor.execute(table_query).fetchall()

    if len(tables) <= 2:
        prompt = generate_prompt(tables, cursor, limit_value, normalization, prompt_db)
    else:
        # get all the foreign keys
        foreign_keys, referenced_tables = get_foreign_key_info(tables, cursor)

        prompt = ""
        included_tables = 0
        for table in tables:
            table_name = table[1]
            if normalization:
                table_name = table_name.lower()

            if included_tables < 2 or (table_name in foreign_keys and foreign_keys[table_name]) or (table_name in referenced_tables):
                prompt += generate_table_prompt(table, cursor, limit_value, normalization, prompt_db)
                included_tables += 1

    return prompt

def prompt_generate_forwiki_multi(db_id,NLQ,SQL,prompt_db="CreateTableSelectCol",task_id=0):
    table_path = os.path.join("../data/wikisql/tables_made",f"task_{task_id}_db",f"{db_id}.sql")
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    with open(table_path, 'r') as sql_file:
        sql_script = sql_file.read()
    cursor.executescript(sql_script)

    limit_value = 0
    prompt = extract_create_table_prompt_forwiki(prompt_db=prompt_db, db_id=db_id,cursor=cursor,limit_value=limit_value, normalization=True)
    conn.close()
    
    prompt = "[INST]" + prompt + "Natural Language Question: "+NLQ + '\n' + "SQL query: "+ SQL + "\n\n" + \
        "--Do the above natural language questions correspond to the SQL queries? Please answer only yes or no, without any explanation. If no, please provide the correct SQL statements." + "[/INST]"
    '''
    prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>" + prompt + "Natural Language Question: "+NLQ + '\n' + "SQL query: "+ SQL + "\n\n" + \
        "--Do the above natural language questions correspond to the SQL queries? Please answer only yes or no, without any explanation. If no, please provide the correct SQL statements." + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    '''
    '''
    prompt_len = get_prompt_length(prompt)
    print("prompt length:", prompt_len)
    '''
    return prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--db_id', help='db_id in spider-dev or spider-train', choices=spider_dev_db_ids + spider_train_db_ids, default="network_1")
    supported_db_prompts = ["Table(Columns)", "Columns=[]", "Columns=[]+FK", "CreateTable", "CreateTableInsertRow", "CreateTableSelectRow",
                            "CreateTableSelectCol"]
    parser.add_argument('--prompt_db', default="CreateTableSelectCol", type=str, choices=supported_db_prompts, help='prompt for db')

    args = parser.parse_args()
    
    prompt = prompt_generate(args.db_id,args.prompt_db,"try")