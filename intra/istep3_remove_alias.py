from records import Database
from typing import List, Dict, Union
from collections import namedtuple
from fuzzywuzzy import process
from mo_sql_parsing import parse_mysql, format
from tqdm import tqdm
import os
import pandas as pd
import sqlite3
import concurrent
import argparse

FromClause = namedtuple("FromClause", ["names", "links"], defaults=[[], {}])
AbstractSyntaxTree = Union[str, dict, list, int, bool, float]

input_root_path = "../data/intra_data/step2_out"
output_root_path = "../data/intra_data/step2_out"
database_path = "../data/spider/database"

parser = argparse.ArgumentParser()
parser.add_argument('--task_id',default=0, type=int)
args = parser.parse_args()

task_id = args.task_id
input_path = os.path.join(input_root_path,"task_" + str(task_id) + ".json")
output_path = os.path.join(output_root_path,"task_" + str(task_id) + ".json")



timeout = 10



class DatabaseAnalyzer(object):
    def __init__(self, db: Database) -> None:
        self._db = db

    @property
    def database(self) -> Database:
        return self._db

    @property
    def tables(self) -> List[str]:
        if not hasattr(self, "_tables"):
            self._tables = [x["name"].lower() for x in self._db.query("SELECT name FROM sqlite_master WHERE type = 'table'").as_dict()]
        return self._tables

    def analyze_columns(self) -> None:
        if not hasattr(self, "_columns"):
            self._columns: Dict[str, dict] = {}
            for table in self.tables:
                self._columns[table] = self._db.query(f"PRAGMA table_info({table})").as_dict()

    @property
    def columns(self) -> Dict[str, List[str]]:
        if not hasattr(self, "_column_names"):
            self.analyze_columns()
            self._column_names = {}
            for table in self.tables:
                self._column_names[table] = [x["name"].lower() for x in self._columns[table]]
        return self._column_names

def fuzzy_name_match(name: str, choices: List[str]) -> str:
    """
    ---
    name : str
    tables : List[str]

    return 
    -----
    matched : str
    """
    return process.extractOne(name, choices)[0]


def analyze_from_clause(ast: AbstractSyntaxTree, da: DatabaseAnalyzer) -> FromClause:
    """
    return 
    -----
    table_info : FromClause
        names : list
        links : dict
    """
    names = []
    links = {}
    if isinstance(ast, str):
        names.append(fuzzy_name_match(ast, da.tables))
    elif isinstance(ast, dict):
        if "value" in ast:
            name = fuzzy_name_match(ast["value"], da.tables)
            alia = ast["name"]
            names.append(name)
            links[alia] = name
    elif isinstance(ast, list):
        for subtree in ast:
            if isinstance(subtree, dict):
                if "join" in subtree:
                    subtree = subtree["join"]
            if isinstance(subtree, str):
                names.append(fuzzy_name_match(subtree, da.tables))
            else:
                name = fuzzy_name_match(subtree["value"], da.tables)
                alia = subtree["name"]
                names.append(name)
                links[alia] = name
    return FromClause(names, links)


def attach_table_name(ast: AbstractSyntaxTree, da: DatabaseAnalyzer,
                      table_info: FromClause=FromClause()) -> AbstractSyntaxTree:
    """
    return 
    -----
    new_ast : AbstractSyntaxTree
    """
    if isinstance(ast, list):
        return [attach_table_name(subtree, da, table_info) for subtree in ast]
    if isinstance(ast, dict):
        if "name" in ast:
            '''
            print("now\n")
            print(table_info)
            print(ast)
            print(ast["name"])
            print("\n\n\n")
            '''
            return table_info.links[ast["name"]]
        if "from" in ast:
            curr_table_info = analyze_from_clause(ast["from"], da)
            merged_dict = {**table_info.links, **curr_table_info.links}
            table_info = FromClause(curr_table_info.names + table_info.names,
                                    merged_dict)
        d = {}
        for key, value in ast.items():
            if key == "literal" or key == "sort":
                d[key] = value
            elif key == "from":
                if isinstance(value, str):
                    d[key] = fuzzy_name_match(value, table_info.names)
                elif isinstance(value, list):
                    if isinstance(value[0], str):
                        d[key] = [fuzzy_name_match(value[0], table_info.names)]
                    else:
                        d[key] = [table_info.links[value[0]["name"]]]
                    d[key] += [attach_table_name(i, da, table_info) for i in value[1:]]
                else:
                    d[key] = attach_table_name(value, da, table_info)
            elif key == "join":
                if isinstance(value, str):
                    d[key] = fuzzy_name_match(value, table_info.names)
                else:
                    d[key] = table_info.links[value["name"]]
            else:
                d[key] = attach_table_name(value, da, table_info)
        return d
    if isinstance(ast, str) and ast != "*":
        if "." in ast:
            table, column = ast.split(".")
            if table in table_info.links:
                table = table_info.links[table]
                column = fuzzy_name_match(column, da.columns[table])
                return f"{table}.{column}"
    return ast


def f(da, query):
    ast = parse_mysql(query)
    ast = attach_table_name(ast, da)
    return format(ast)



def execute_SQL(SQL,file_path):
    with open(file_path,'r') as f:
        conn = sqlite3.connect(f.name)

        df = pd.read_sql_query(f"{SQL}",conn)

        conn.close()
    exe_result = [item[0] for item in df.values.tolist()]
    if len(exe_result) == 0:
        return (exe_result , False)
    elif len(exe_result) == 1 and exe_result[0] is None:
        return (exe_result , False)
    else:
        return (exe_result , True)

def sql_execute(db_id,sql):
    db_path = os.path.join(database_path,db_id,db_id+".sqlite")
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(execute_SQL, sql, db_path)
            try:
                result = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                print("time out")
                result = ([],False)
        #if execute_SQL(SQL,file_path):
            return result
    except:
        return ([],False)

import json

if __name__ == "__main__":
    with open(input_path, "r") as fp:
        train = json.load(fp)
    res = []
    for sample in tqdm(train):
        db_id = sample["db_id"]
        question = sample["question"]
        query = sample["query"]
        da = DatabaseAnalyzer(Database(f"sqlite:///../data/spider/database/{db_id}/{db_id}.sqlite"))

        result_1 = sql_execute(db_id,query)
        result_2 = sql_execute(db_id,f(da, query))

        if result_1[1] is True and result_2[1] is True:
            if result_1[0] == result_2[0]:
                res.append({
                    "db_id": db_id,
                    "question": question,
                    "query": f(da, query)
                    })
            else:
                print(result_1,result_2)
        else:
            print(result_1,result_2)
    with open(output_path, "w") as fp:
        json.dump(res,fp)