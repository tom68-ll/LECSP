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
database_path = "../data/spider/database"
if args.order == 0:
    input_path = "../data/only_task0_order0/Mistral/step0_out"
    output_path = '../data/only_task0_order0/Mistral/step1_out'
    data_cluster_path = "../data/cluster_result/cluster"
    db_id_path = "../data/combine1_perm_1"
else:
    input_path = "./data/inter_data" + str(args.order) + "/step0_out"
    output_path = './data/inter_data' + str(args.order) + '/step1_out'
    data_cluster_path = "../data/cluster_result/cluster" + str(args.order)
    db_id_path = "../data/spider_task_stream_order" + str(args.order)


timeout = 10
cnt = 0

def extract_nlq_sql_pairs(text):
    lines = text

    nlq_sql_pairs = []
    current_nlq = ""
    current_sql = ""
    in_nlq = False  # True if processing NLQ

    for line in lines:
        if re.match(r"--(\d+\.|Question \d+:)", line) or (in_nlq and line.startswith('--')):
            # check if its the start of a SQL
            if current_nlq and current_sql:
                nlq_sql_pairs.append((current_nlq.strip(), current_sql))
                current_nlq, current_sql = "", ""
            in_nlq = True
            # clear signs in NLQ
            cleaned_line = re.sub(r"--(\d+\.|Question \d+:) ?", "", line).strip()
            current_nlq += " " + cleaned_line
        elif in_nlq and line.strip() and not line.startswith('--'):
            current_sql = line.strip()
            in_nlq = False 
        elif line.strip() == "" and current_nlq and current_sql:
            # blank line is the end of NLQ-SQL pairs
            nlq_sql_pairs.append((current_nlq.strip(), current_sql))
            current_nlq, current_sql = "", ""

    # add the last NLQ-SQL pairs (if exists)
    if current_nlq and current_sql:
        nlq_sql_pairs.append((current_nlq.strip(), current_sql))

    return nlq_sql_pairs

def new_extract_nlq_sql_pairs(text):
    # split the text
    text = "".join(text)
    segments = re.split(r'\d+\.', text)
    pairs = []

    for segment in segments:
        if segment.strip():
            # extract NLQ
            nlq = ""
            if "Question:" in segment or "question:" in segment:
                # start extract from Question: or question:
                nlq_match = re.search(r'(Question:|question:)(.*?)(, query|\{query)', segment, re.DOTALL)
                if nlq_match:
                    nlq = nlq_match.group(2).strip()
                else:
                    # if not exists ', query' or '{query', then check'}\n' and '\n'
                    nlq_match = re.search(r'(Question:|question:)(.*?)(}\n|\n)', segment, re.DOTALL)
                    if nlq_match:
                        nlq = nlq_match.group(2).strip()
            else:
                nlq_match = re.search(r'^(.*?)\n', segment, re.DOTALL)
                if nlq_match:
                    nlq = nlq_match.group(1).strip()

            # extract SQL
            sql = ""
            sql_match = re.search(r'(SELECT.*?);', segment, re.DOTALL)
            if sql_match:
                sql = sql_match.group(1).strip() + ';'
            else:
                for ending in ['" }\n', '"}\n', ' }\n', '}\n', '```']:
                    sql_match = re.search(r'(SELECT.*?)' + re.escape(ending), segment, re.DOTALL)
                    if sql_match:
                        sql = sql_match.group(1).strip() + ';'
                        break

            if nlq and sql:
                pairs.append((nlq.replace("\\","").replace("\n"," "),\
                               sql.replace("\\","").replace("\n"," ")))

    return pairs






def extract_sql_keywords_after_prefix(text_line):
    prefix = "--Create 10 suitable SQL queries that simultaneously include the SQL keywords: "
    
    if prefix in text_line:
        keywords_part = text_line.split(prefix)[-1].split(". And")[0]

        keywords = keywords_part.split()

        keywords_dict = {keyword.strip('.,'): True for keyword in keywords}

        return keywords_dict
    else:
        return {}
    

def score_sql(sql, keywords):
    score = 0

    for keyword in keywords:
        if keyword in sql:
            score += 1

    if '{DASC}' in keywords and re.search(r'\bDESC\b|\bASC\b', sql):
        score += 1

    if '{OP}' in keywords and re.search(r'\b(=|<>|>|<|>=|<=|BETWEEN|IN|LIKE|AND|OR|NOT)\b', sql):
        score += 1

    if '{NESTING}' in keywords and sql.count('SELECT') > 1:
        score += 1


    return score

'''
def db_id_reader(task_id):

    task_test_pth = os.path.join(db_id_path, f"task_{task_id}", "train.json")

    db_id_set = set()

    with open(task_test_pth, 'r') as f1:
        test_data = json.load(f1)
        for dt in test_data:
            db_id_set.add(dt["db_id"])
    return db_id_set
'''

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
    
def can_be_execute_forwiki_multi(db_id,pairs,task_id):
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
    table_path = os.path.join('/home/jyzhang/user1/wikisql/tables_made',f"task_{task_id}_db",f"{db_id}.sql")
    with open(table_path, 'r') as sql_file:
        sql_script = sql_file.read()
    cursor.executescript(sql_script)
    conn.commit()
    output_tmp = []
    for NLQ,SQL in pairs:
        SQL_new = SQL
        results = execute_query(conn, SQL_new)
        if results:
            output_tmp.append((NLQ,SQL))
    return output_tmp
    
def can_be_execute_forwiki(db_id,pairs):
    wikitable_dir = '/home/jyzhang/user1/wikisql/tables.jsonl'
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

    output_tmp = []
    db_conn = create_database(table_now,table_name_new,headers_new)
    for NLQ,SQL in pairs:
        SQL_new = SQL
        for replace_key in list(replace_dict.keys()):
            SQL_new = re.sub(replace_key,replace_dict[replace_key],SQL_new)
        results = execute_query(db_conn, SQL_new)
        if results:
            output_tmp.append((NLQ,SQL))

    db_conn.close()
    return output_tmp

def can_be_execute(db_id,sql):
    if sql == "SELECT DISTINCT alid, name, iata, icao, callsign, country, active FROM airlines a WHERE alid IN (     SELECT alid FROM routes r     WHERE src_apid IN (         SELECT apid FROM airports         WHERE country = (             SELECT country FROM airports WHERE apid = r.dst_apid         )     )     AND dst_apid IN (         SELECT apid FROM airports         WHERE country = (             SELECT country FROM airports WHERE apid = r.src_apid         )     ) ) AND icao IS NOT NULL;":
        return False
    db_path = os.path.join(database_path,db_id,db_id+".sqlite")
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(execute_SQL, sql, db_path)
            try:
                result = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                print("time out")
                return False
        #if execute_SQL(SQL,file_path):
            return result
    except:
        return False



def contains_letter(s):
    return any(char.isalpha() for char in s)

def judge_to_execute(pairs,db_id_now,name_order,task_id):
    if name_order == 'spider':
        output_pairs = []

        for pair in pairs:
            can_be_ex = can_be_execute(db_id_now,list(pair)[1])
            if can_be_ex:
                output_pairs.append(pair)
    elif name_order == 'wikisql':
        if contains_letter(db_id_now):
            output_pairs = []
            output_pairs = can_be_execute_forwiki_multi(db_id_now,pairs,task_id)
        else:
            output_pairs = []
            output_pairs = can_be_execute_forwiki(db_id_now,pairs)
                
    return output_pairs


def sort_sql_statements(nlq_sql_tuples, keywords):
    scored_sql = [(nlq_sql, score_sql(nlq_sql[1], keywords)) for nlq_sql in nlq_sql_tuples]

    scored_sql.sort(key=lambda x: x[1], reverse=True)

    sorted_nlq_sql = [item[0] for item in scored_sql]

    return sorted_nlq_sql

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
    name_orders = ['spider','wikisql','spider','wikisql','spider','wikisql','spider']
    #name_orders = ['wikisql','spider','spider','spider','spider','wikisql','wikisql']
    
    get_label_root = "./info_for_step1_former/label_of"
    task_rate = [1,2,4,2,2,2,2]
    for task_now in range(7):
        if task_now % 2 == 0:
            get_text = True
        else:
            get_text = False
        #task_now = args.task_id
        name_order = name_orders[task_now]

        file_path = os.path.join(input_path,"task_" + str(task_now) + ".txt")
        out_path = os.path.join(output_path,"task_" + str(task_now) + ".json")

        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        i = 0
        content = []
        output_nlq = []
        output_sql = []
        output_dbid = []
        output_text = []
        output_label = []
        
        with open(get_label_root + str(task_now) + ".json" , 'r') as fp:
            label_infos = json.load(fp)

        start_toread = False
        key_words = {}
        sql_set_readed = 0
        db_id_now = False
        discount = 0
        counting = 0
        pbar = tqdm(total=len(lines))
        while i < len(lines):
            
            if '***********************************************' in lines[i]:
                if start_toread == True:
                    start_toread = False
                    pairs = new_extract_nlq_sql_pairs(content)
                    if len(pairs) == 0:
                        discount += 1
                        
                    old_pairs = pairs
                    pairs = judge_to_execute(pairs, db_id_now,name_order,task_now)
                    pairs = sort_sql_statements(pairs, key_words)
                    counting += len(pairs)
                    for nlq, sql in pairs[:task_rate[task_now]]:
                        if len(nlq.split('?')) > 1:
                            output_nlq.append(nlq.split('?')[0] + '?')
                        else:
                            output_nlq.append(nlq)
                        output_sql.append(sql)
                        output_dbid.append(db_id_now)
                        output_label.append(label_infos[sql_set_readed])
                        output_text.append("")
                    sql_set_readed += 1
                db_id_now = lines[i+1].split(":")[1].strip()

            if start_toread == True:
                content.append(lines[i])
                
            if i > 0 and 'generated_text_start:' in lines[i]:
                start_toread = True
                content = []
                key_words = extract_sql_keywords_after_prefix(lines[i-5])
            i += 1
            pbar.update(1)
        pairs = new_extract_nlq_sql_pairs(content)
        pairs = judge_to_execute(pairs, db_id_now,name_order,task_now)
        pairs = sort_sql_statements(pairs, key_words)
        counting += len(pairs)
        for nlq, sql in pairs[:task_rate[task_now]]:
            output_nlq.append(nlq)
            output_sql.append(sql)
            output_dbid.append(db_id_now)
            output_label.append(label_infos[sql_set_readed])
            output_text.append("")
        sql_set_readed += 1

        print("counting: {}".format(counting))
        print("chosen: {}".format(len(output_nlq)))

        output_tmp = []
        for i in range(len(output_nlq)):
            output_tmp.append({
                "db_id": output_dbid[i],
                "question": output_nlq[i],
                "query": output_sql[i],
                "text" : output_text[i],
                "label": output_label[i]
            })

        output = []
        set_of_deduplication = set()
        for item_dict in output_tmp:
            if item_dict["question"] + "||" + item_dict["query"] in set_of_deduplication:
                continue
            set_of_deduplication.add(item_dict["question"] + "||" + item_dict["query"])
            if "||" in item_dict["query"]:
                continue
            
            replacer = SQLAliasReplacer(item_dict["query"])
            new_sql_query = replacer.replace_aliases()

            output.append({
                "db_id": item_dict["db_id"],
                "question": item_dict["question"],
                "query": new_sql_query,
                "text": item_dict["text"],
                "label": item_dict["label"]
            })

        with open(out_path,'w') as f:
            json.dump(output,f)

        print("sql_set_readed : {}".format(sql_set_readed))
        print("discount : {}".format(discount))