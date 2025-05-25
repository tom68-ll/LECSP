import re
import json
import os
import concurrent.futures
import sqlite3
import pandas as pd

import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML
import argparse
import jsonlines
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', type=int,default=0)
parser.add_argument('--order', type=int,default=0,choices=[0,1])
args = parser.parse_args()

# to choose the order of data input
if args.order == 0:
    input_path1 = "../data/only_task0_order0/Mistral/step2_out"
    input_path2 = "../data/only_task0_order0/Mistral/step1_out"
    output_path = '../data/only_task0_order0/Mistral/step3_out'

    data_cluster_path = "../data/cluster_result"
    database_path = "../data/spider/database"
else:
    input_path1 = "../data/inter_data" + str(args.order) + "step2_out"
    input_path2 = "../data/inter_data" + str(args.order) + "step1_out"
    output_path = "../data/inter_data" + str(args.order) + "step3_out"

    data_cluster_path = "../data/cluster_result" + str(args.order)
    database_path = "../data/spider/database"

timeout = 10
cnt = 0

def search_SQL(text):
    text = "".join(text)
    sql = ""
    sql_match = re.search(r'(SELECT.*?);', text, re.DOTALL)
    if sql_match:
        sql = sql_match.group(1).strip() + ';'
    sql = sql.replace("\_","_").replace("\n"," ").replace("\*","*")
    return sql

def load_json(json_file) -> list:
    with open(json_file, 'r', encoding='utf-8') as f:
        ex_list = json.load(f)
    return ex_list

def calculate_union(now_idx):
    centers = []
    for idx in range(now_idx):
        path_of_this = os.path.join(data_cluster_path,"task_"+str(idx)+'.json')
        datai = load_json(path_of_this)
        for data in datai:
            centers.append(frozenset(data["center"]))
    return set(centers)

def calculate_now(now_idx):
    centers = []
    path_of_this = os.path.join(data_cluster_path,"task_"+str(now_idx)+'.json')
    datai = load_json(path_of_this)
    for data in datai:
        centers.append(frozenset(data["center"]))
    return set(centers)

def calculate_diff(set1,set2):
    return set1 - set2


def read_pattern_number(task_id):
    if task_id != 0:
        pattern_before = calculate_union(task_id)
        pattern_now = calculate_now(task_id)
        pattern_to_use = calculate_diff(pattern_before,pattern_now)
    return len(pattern_to_use)

def execute_SQL(SQL,file_path):
    with open(file_path,'r') as f:
        conn = sqlite3.connect(f.name)

        df = pd.read_sql_query(f"{SQL}",conn)

        conn.close()
    exe_result = [item[0] for item in df.values.tolist()]
    if len(exe_result) == 0:
        return False
    elif len(exe_result) == 1 and exe_result[0] is None:
        return False
    else:
        return True

def can_be_execute_forwiki_multi(db_id,sql,task_id):
    def execute_query(conn, query):
        cursor = conn.cursor()
        try:
            cursor.execute(query)
            results = cursor.fetchall()
        except sqlite3.OperationalError as e:
            print(f"SQLite error: {e}")
            results = None
        except Exception as e:
            print(f"Error: {e}")
            results = None

        return results
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    table_path = os.path.join('../data/wikisql/tables_made',f"task_{task_id}_db",f"{db_id}.sql")
    with open(table_path, 'r') as sql_file:
        sql_script = sql_file.read()
    cursor.executescript(sql_script)
    conn.commit()
    output_tmp = []
    SQL_new = sql
    results = execute_query(conn, SQL_new)
    if results:
        return True
    else:
        return False
    
def can_be_execute_forwiki(db_id,sql):
    wikitable_dir = '../data/wikisql/tables.jsonl'
    def create_database(data,table_name,headers):
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()

        headers = [item.replace(' ','_') for item in headers]
        columns = ', '.join([f'"{header}"' for header in headers])
        cursor.execute(f"CREATE TABLE {table_name} ({columns});")

        for row in data['rows']:
            cursor.execute(f"INSERT INTO {table_name} VALUES ({', '.join(['?'] * len(row))});", row)

        conn.commit()
        return conn

    def execute_query(conn, query):
        cursor = conn.cursor()
        try:
            cursor.execute(query)
            results = cursor.fetchall()
        except sqlite3.OperationalError as e:
            print(f"SQLite error: {e}")
            results = None
        except Exception as e:
            print(f"Error: {e}")
            results = None

        return results
    table_now = {}
    with open(wikitable_dir,'r') as fp:
        for item in jsonlines.Reader(fp):
            if item['id'] == db_id:
                table_now = item
            else:
                continue

    if not table_now:
        raise NotImplemented("dont have the table")

    table_name = table_now['id']
    table_name_new = 'table_' + re.sub('-','_',table_now['id'])
    headers = table_now['header']
    headers_new = [item.replace('/','_') for item in headers]

    replace_dict = {}
    replace_dict[table_name] = table_name_new
    for former,later in zip(headers,headers_new):
        replace_dict[former] = later

    db_conn = create_database(table_now,table_name_new,headers_new)
    SQL_new = sql
    for replace_key in list(replace_dict.keys()):
        SQL_new = re.sub(replace_key,replace_dict[replace_key],SQL_new)
    results = execute_query(db_conn, SQL_new)

    db_conn.close()
    return results

def contains_letter(s):
    return any(char.isalpha() for char in s)

def can_be_execute(db_id,sql,name_order,task_id):
    if name_order == 'spider':
        db_path = os.path.join(database_path,db_id,db_id+".sqlite")
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(execute_SQL, sql, db_path)
                try:
                    result = future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    print("time out")
                    result = False
            #if execute_SQL(SQL,file_path):
                return result
        except:
            return False
    elif name_order == 'wikisql':
        if contains_letter(db_id):
            return can_be_execute_forwiki_multi(db_id,sql,task_id)
        else:
            return can_be_execute_forwiki(db_id,sql)

def judge_to_execute(pair,db_id_now):

    can_be_ex = can_be_execute(db_id_now,pair)
    return can_be_ex

class SQLAliasReplacer:
    def __init__(self, sql):
        self.sql = sql

    @staticmethod
    def is_subselect(parsed):
        if not parsed.is_group:
            return False
        for item in parsed.tokens:
            if item.ttype is DML and item.value.upper() == 'SELECT':
                return True
        return False

    @staticmethod
    def extract_from_part(parsed):
        from_seen = False
        for item in parsed.tokens:
            if from_seen:
                if SQLAliasReplacer.is_subselect(item):
                    for x in SQLAliasReplacer.extract_from_part(item):
                        yield x
                elif item.ttype is Keyword and item.value.upper() in ['ORDER', 'GROUP', 'BY', 'HAVING']:
                    from_seen = False
                    StopIteration
                else:
                    yield item
            elif item.ttype is Keyword and item.value.upper() == 'FROM':
                from_seen = True

    @staticmethod
    def extract_table_identifiers(token_stream):
        for item in token_stream:
            if isinstance(item, IdentifierList):
                for identifier in item.get_identifiers():
                    yield identifier
            elif isinstance(item, Identifier):
                yield item
            elif item.ttype is Keyword:
                return

    def replace_aliases(self):
        parsed = sqlparse.parse(self.sql)[0]
        for item in self.extract_from_part(parsed):
            if isinstance(item, Identifier):
                real_name = item.get_real_name()
                alias = item.get_alias()
                if alias and real_name != alias:
                    self.sql = self.sql.replace(f' {real_name} AS {alias}', f' {real_name}')
                    self.sql = self.sql.replace(f' {alias}.', f' {real_name}.')
                    self.sql = self.sql.replace(f' {alias} ', f' {real_name} ')

        return self.sql

if __name__ == '__main__':
    #task_now = args.task_id
    name_orders = ['spider','wikisql','spider','wikisql','spider','wikisql','spider']
    #name_orders = ['wikisql','spider','spider','spider','spider','wikisql','wikisql']
    for task_now in range(7):
        name_order = name_orders[task_now]
        file_path = os.path.join(input_path1,"task_" + str(task_now) + ".json")
        original_data_path = os.path.join(input_path2,"task_" + str(task_now) + ".json")
        out_path = os.path.join(output_path,"task_" + str(task_now) + ".json")

        with open(file_path, 'r', encoding='utf-8') as file:
            corrections = json.load(file)

        with open(original_data_path,'r',encoding='utf-8') as file:
            original_data = json.load(file)


        output_tmp = []
        for correction in tqdm(corrections):
            if correction['result'].strip().lower().startswith('yes'):
                output_tmp.append({
                    "db_id" : correction["db_name"],
                    "question" : correction["question"],
                    "query" : correction["gold_sql"],
                    "text" : correction["text"],
                    "label" : correction["label"]
                })
            elif correction['result'].strip().lower().startswith('no'):
                tmp_sql = search_SQL(correction['result'])
                if can_be_execute(correction["db_name"],tmp_sql,name_order,task_now):
                    output_tmp.append({
                        "db_id" : correction["db_name"],
                        "question" : correction["question"],
                        "query" : tmp_sql,
                        "text" : correction["text"],
                        "label" : correction["label"]
                    })
                else:
                    print("cant be execute")
                    output_tmp.append({
                    "db_id" : correction["db_name"],
                    "question" : correction["question"],
                    "query" : correction["gold_sql"],
                    "text" : correction["text"],
                    "label" : correction["label"]
                })
            else:
                print("*****" + correction['result'])
                raise ValueError("start with not Yes and not No")


        output = []
        set_of_deduplication = set()
        #make process to the extracted data
        for item_dict in tqdm(output_tmp):
            if item_dict["question"] + "||" + item_dict["query"] in set_of_deduplication:
                continue
            set_of_deduplication.add(item_dict["question"] + "||" + item_dict["query"])
            #delete if contains '||'
            if "||" in item_dict["query"]:
                continue
            
            #remove alias
            replacer = SQLAliasReplacer(item_dict["query"])
            new_sql_query = replacer.replace_aliases()

            #output
            output.append({
                "db_id": item_dict["db_id"],
                "question": item_dict["question"],
                "query": new_sql_query,
                "text" : item_dict["text"],
                "label" : item_dict["label"]
            })

        with open(out_path,'w') as f:
            json.dump(output,f)