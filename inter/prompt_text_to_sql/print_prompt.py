import argparse
#from database_prompt_construction import generate_db_prompt
from prompt_text_to_sql.database_prompt_construction import generate_db_prompt
#from sql_generation import get_prompt_length
#from utils import spider_train_db_ids, spider_dev_db_ids
from prompt_text_to_sql.utils import spider_train_db_ids, spider_dev_db_ids
import os
import sqlite3


def prompt_generate(db_id,keywords_set,prompt_db="CreateTableSelectCol",prompt_type="ordinary"):
    prompt_db = prompt_db
    db_id = db_id
    
    if db_id in spider_dev_db_ids:
        dataset = "spider-dev"
    elif db_id in spider_train_db_ids:
        dataset = "spider-train"
    else:
        dataset = "wikisql"
    

    if prompt_db in ["Table(Columns)", "Columns=[]", "Columns=[]+FK", "CreateTable"]:
        limit_value = 0
    elif prompt_db in ["CreateTableInsertRow", "CreateTableSelectRow", "CreateTableSelectCol"]:
        limit_value = 1
    else:
        raise ValueError("Unknown prompt_db")
    prompt_length_by_db = {}
    prompt = generate_db_prompt(dataset, db_id, prompt_db=prompt_db, limit_value=limit_value, normalization=True)

    if prompt_type == "ordinary":
        prompt = prompt  + "-- Using valid SQLite, answer the following questions for the tables provided above.\n"
        prompt = prompt + '\n' + "--Using valid SQLite, generate 10 different questions their corresponding SQL statements, and the execution results correctly. "\
            + '\n' +  '--The generated SQL statements, in addition to the basic SQL syntax, should also include the following keywords:'\
            + ' '.join(keywords_set) + '.' +'\n\n' + '--If "{NESTING}" is in the keyword, it means that the generated SQL should include nested queries.If "{OP}" is in the keyword, it means that the generated SQL should include operators, such as >, < and ='
    elif prompt_type == "no_OP":
        keywords_set = set(keywords_set)
        # remove "{OP}" and "{NESTING}"
        keywords_set.discard("{OP}")
        keywords_set.discard("{NESTING}")
        prompt = prompt  + "-- Using valid SQLite, answer the following questions for the tables provided above.\n"
        prompt = "[INST]" + prompt + '\n' + "--Using valid SQLite, generate 10 different questions their corresponding SQL statements, and the execution results correctly. "\
            + '\n' +  '--The generated SQL statements, in addition to the basic SQL syntax, should also include the following keywords:'\
            + ' '.join(keywords_set) + '.' +'[/INST]' + '\n'
    elif prompt_type == "V1":
        keywords_set = list(keywords_set)
        # remove "{OP}" and "{NESTING}"
        #mixtral
        
        prompt = "[INST]" + prompt +  "--Create 10 suitable SQL queries that simultaneously include the SQL keywords: " + ' '.join(keywords_set) + \
            ". And provide a corresponding and accurate natural language question for each query." + "\n"\
            + "--The natural language question must accurately reflect the intent of the SQL query." + "\n"\
            + "--Return the results in the format of {question:Question, query:Query}" + '\n'\
            + '--If "{NESTING}" is in the keyword, it means that the generated SQL should include nested queries.If "{OP}" is in the keyword, it means that the generated SQL should include operators, such as >, < and =' + '[/INST]' + '\n'
        

        #Llama
        '''
        prompt =  "<|begin_of_text|><|start_header_id|>user<|end_header_id|>" + prompt +  "--Create 10 suitable SQL queries that simultaneously include the SQL keywords: " + ' '.join(keywords_set) + \
            ". And provide a corresponding and accurate natural language question for each query." + "\n"\
            + "--The natural language question must accurately reflect the intent of the SQL query." + "\n"\
            + "--Return the results in the format of {question:Question, query:Query}" +  '\n'\
            + '--If "{NESTING}" is in the keyword, it means that the generated SQL should include nested queries.If "{OP}" is in the keyword, it means that the generated SQL should include operators, such as >, < and =' + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        '''
        #Qwen
        '''
        prompt =  prompt +  "--Create 10 suitable SQL queries that simultaneously include the SQL keywords: " + ' '.join(keywords_set) + \
            ". And provide a corresponding and accurate natural language question for each query." + "\n"\
            + "--The natural language question must accurately reflect the intent of the SQL query." + "\n"\
            + "--Return the results in the format of {question:Question, query:Query}" +  '\n'\
            + '--If "{NESTING}" is in the keyword, it means that the generated SQL should include nested queries.If "{OP}" is in the keyword, it means that the generated SQL should include operators, such as >, < and =' + '\n'
        '''
                
        
    '''
    prompt_len = get_prompt_length(prompt)
    print("prompt length:", prompt_len)
    '''
    return prompt


def generate_data_prompt(headers, top_k_rows, table_name, limit_value, normalization, prompt_db):
    data_prompt = ""
    if limit_value > 0:
        if prompt_db.startswith("CreateTableSelectRow"):
            data_prompt += f"/*\n3 example rows:\nSELECT * FROM {table_name} LIMIT {limit_value};\n{'    '.join(headers)}\n"
            for row in top_k_rows:
                row = [str(x) for x in row]
                row = [x if x is not None else "" for x in row]

                data_prompt += '    '.join(row) + "\n"
            data_prompt += "*/\n"
        elif prompt_db.startswith("CreateTableInsertRow"):
            for row in top_k_rows:
                insert_statement = "INSERT INTO " + table_name + " ("
                insert_statement += ', '.join(headers) + ") VALUES "
                row = [x if x is not None else "" for x in row]
                row = [str(x) if isinstance(x, (int, float)) else '"' + str(x) + '"' for x in row]
                insert_statement += "(" + ', '.join(row) + ");"
                data_prompt += insert_statement + "\n"
    return data_prompt

def generate_table_prompt(table, cursor, limit_value, normalization, prompt_db):
    table_prompt = ""
    table_name = table[1]
    if normalization:
        table_name = table_name.lower()

    create_table_statement = table[-1]
    table_info_query = f"PRAGMA table_info({table_name});"
    top_k_row_query = f"SELECT * FROM {table_name} LIMIT {limit_value};"

    headers = [x[1] for x in cursor.execute(table_info_query).fetchall()]
    if normalization:
        create_table_statement = create_table_statement.lower()
        top_k_row_query = top_k_row_query.lower()
        headers = [x.lower() for x in headers]
    top_k_rows = cursor.execute(top_k_row_query).fetchall()

    table_prompt += create_table_statement + ";\n"
    table_prompt += generate_data_prompt(headers, top_k_rows, table_name, limit_value, normalization, prompt_db) + "\n"
    return table_prompt

def get_foreign_key_info(tables, cursor):
    foreign_keys = {}
    referenced_tables = set()
    for table in tables:
        table_name = table[1]
        fk_query = f"PRAGMA foreign_key_list({table_name});"
        fk_info = cursor.execute(fk_query).fetchall()
        foreign_keys[table_name] = [fk[2] for fk in fk_info]  
        for ref_table in [fk[2] for fk in fk_info]:
            referenced_tables.add(ref_table)
    return foreign_keys, referenced_tables

def generate_prompt(tables, cursor, limit_value, normalization, prompt_db):
    prompt = ""
    for table in tables:
        prompt += generate_table_prompt(table, cursor, limit_value, normalization, prompt_db)
    return prompt

def extract_create_table_prompt(prompt_db, db_id, cursor, limit_value=3, normalization=True):

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


def multi_table_prompt_make(table_path,keywords_set,prompt_db="CreateTableSelectCol",prompt_type="ordinary"):
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    with open(table_path, 'r') as sql_file:
        sql_script = sql_file.read()
    print(sql_script)
    cursor.executescript(sql_script)
        
    schema_prompt = extract_create_table_prompt(prompt_db, None, cursor, limit_value=1, normalization=True)
    if prompt_type == "V1":
        keywords_set = list(keywords_set)
        # remove "{OP}" and "{NESTING}"
        prompt = "[INST]" + schema_prompt +  "--Create 10 suitable SQL queries that simultaneously include the SQL keywords: " + ' '.join(keywords_set) + \
            ". And provide a corresponding and accurate natural language question for each query." + "\n"\
            + "--The natural language question must accurately reflect the intent of the SQL query." + "\n"\
            + "--Return the results in the format of {question:Question, query:Query}" + '\n'\
            + '--If "{NESTING}" is in the keyword, it means that the generated SQL should include nested queries.If "{OP}" is in the keyword, it means that the generated SQL should include operators, such as >, < and =' + '[/INST]' + '\n'

    conn.close()
    
    return prompt


'''
def multi_table_prompt_make(table_path,keywords_set,prompt_db="CreateTableSelectCol",prompt_type="ordinary"):
    with open(table_path, 'r') as sql_file:
        sql_script = sql_file.read()
    
    
        
    schema_prompt = extract_create_table_prompt(prompt_db, None, db_path, limit_value=1, normalization=True)
    if prompt_type == "V1":
        keywords_set = list(keywords_set)
        # remove "{OP}" and "{NESTING}"
        prompt = "[INST]" + schema_prompt +  "--Create 10 suitable SQL queries that simultaneously include the SQL keywords: " + ' '.join(keywords_set) + \
            ". And provide a corresponding and accurate natural language question for each query." + "\n"\
            + "--The natural language question must accurately reflect the intent of the SQL query." + "\n"\
            + "--Return the results in the format of {question:Question, query:Query}" + '\n'\
            + '--If "{NESTING}" is in the keyword, it means that the generated SQL should include nested queries.If "{OP}" is in the keyword, it means that the generated SQL should include operators, such as >, < and =' + '[/INST]' + '\n'

    conn.close()

    if os.path.exists(db_path):
        os.remove(db_path)
    
    return prompt
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--db_id', help='db_id in spider-dev or spider-train', choices=spider_dev_db_ids + spider_train_db_ids, default="network_1")
    supported_db_prompts = ["Table(Columns)", "Columns=[]", "Columns=[]+FK", "CreateTable", "CreateTableInsertRow", "CreateTableSelectRow",
                            "CreateTableSelectCol"]
    parser.add_argument('--prompt_db', default="CreateTableSelectCol", type=str, choices=supported_db_prompts, help='prompt for db')

    args = parser.parse_args()
    
    prompt = prompt_generate(args.db_id,args.prompt_db,"try")