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

input_path1 = "../data/intra_data/step1_out"
output_path = '../data/intra_data/step1_out'

database_path = "../data/spider/database"

parser = argparse.ArgumentParser()
parser.add_argument('--task_id', default=0,type=int)
args = parser.parse_args()


task_now = args.task_id

file_path = os.path.join(input_path1,"task_" + str(task_now) + ".json")
out_path = os.path.join(output_path,"task_" + str(task_now) + ".json")
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

ppp = 0
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

def can_be_execute(db_id,sql):
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

    with open(file_path, 'r', encoding='utf-8') as file:
        prompt_output = json.load(file)
    
    output_data = []

    for prompt in prompt_output:
        splited_prompt = prompt.split("####")
        db_id = splited_prompt[1]
        query = splited_prompt[2]
        contain_question = splited_prompt[3]
        answer = contain_question.split("[/INST]\n")[1]
        question = answer.split('\n')[0]
        tokens = question.split()
        while tokens and not tokens[0][0].isalpha():
            tokens[0] = tokens[0][1:]

            if not tokens[0]:
                tokens.pop(0)
        question = ' '.join(tokens)
    
        if "?" in question:
            question = question.split('?')[0] + "?"
        elif "." in question:
            question = question.split('.')[0] + "."
        output_data.append({
            "db_id": db_id,
            "question": question,
            "query": query
        })
    
    with open(out_path,'w') as f:
        json.dump(output_data,f)

