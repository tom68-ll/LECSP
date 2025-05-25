import os
import re
import json
import pickle
import random
from template_config import *
from collections import defaultdict
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import sqlite3
import pandas as pd
import concurrent.futures
import re
import math
import json

import nltk
nltk.download('wordnet')

ps = PorterStemmer()
lmtzr = WordNetLemmatizer()

def read_in_all_data(data_path=DATA_PATH):
    training_data = json.load(open(os.path.join(data_path, "train.json")))
    tables_org = json.load(open(os.path.join(data_path, "tables.json")))
    tables = {tab['db_id']: tab for tab in tables_org}

    return training_data, tables

def get_all_question_query_pairs(data):
    question_query_pairs = []
    for item in data:
        question_query_pairs.append((item['question_toks'], item['query'], item['db_id']))
    return question_query_pairs



def is_value(token):
    """
    as values can either be a numerical digit or a string literal, then we can
    detect if a token is a value by matching with regex
    """
    is_number = True
    try:
        float(token)
    except ValueError:
        is_number = False
    is_string = token.startswith("\"") or token.startswith("\'") or token.endswith("\"") or token.endswith("\'")

    return is_number or is_string


def remove_all_from_clauses(query_keywords):
    """
    remove all keywords from from clauses, until there is no more from clauses
    e.g. select {} from {} as {} where {} = {} --> select {} where {} = {}
    """
    # remove from clause by deleting the range from "FROM" to "WHERE" or "GROUP"
    start_location = 0
    count = 0
    while "FROM" in query_keywords:
        count += 1
        if count > 5:
            break
            print("error query_keywords: ", query_keywords)
        from_location = query_keywords.index("FROM")
        end_token_locations = [len(query_keywords)]  # defaulting to the end of the list
        for end_token in ["WHERE", "GROUP", "ORDER"]:
            try:
                end_token_locations.append(query_keywords.index(end_token, start_location))
            except ValueError:
                pass

        query_keywords = query_keywords[:from_location] + [FROM_SYMBOL] + query_keywords[min(end_token_locations):]
        start_location = min(end_token_locations)
        
    return query_keywords


def strip_query(query, table):
    """
    returns (stripped query, non keywords)
    """
    #get table column names info
    column_types = table['column_types']
    table_names_original = [cn.lower() for cn in table['table_names_original']]
    table_names = [cn.lower() for cn in table['table_names']]
    column_names = [cn.lower() for i, cn in table['column_names']]
    column_names_original = [cn.lower() for i, cn in table['column_names_original']]

    #clean query: replace values, numbers, column names with SYMBOL
    query_keywords = []
    columns = table_names_original + table_names

    query = query.replace(";","")
    query = query.replace("\t","")
    query = query.replace("(", " ( ").replace(")", " ) ")
    # then replace all stuff enclosed by "" with a numerical value to get it marked as {VALUE}
    str_1 = re.findall("\"[^\"]*\"", query)
    str_2 = re.findall("\'[^\']*\'", query)
    
    query_tokenized = query.split(' ')
    float_nums = re.findall("[-+]?\d*\.\d+", query)
    now_idx = 0
    values_dict = {}
    #query_tokenized = [VALUE_NUM_SYMBOL+str(now_idx) if qt in float_nums else qt for qt in query_tokenized]
    query_tmp = []
    for qt in query_tokenized:
        if qt in float_nums:
            values_dict[VALUE_NUM_SYMBOL.split('}')[0]+str(now_idx)] = qt
            query_tmp.append(VALUE_NUM_SYMBOL.split('}')[0]+str(now_idx))
            now_idx += 1
        else:
            query_tmp.append(qt)
    query_tokenized = query_tmp
    
    query_tmp = []
    for token_now in query_tokenized:
        if token_now in values_dict.keys():
            continue
        else:
            query_tmp.append(token_now)
    query = " ".join(query_tmp)
    int_nums = [i.strip() for i in re.findall("[^tT]\d+", query)]

    #query_tokenized = [VALUE_NUM_SYMBOL if qt in int_nums else qt for qt in query_tokenized]
    query_tmp = []
    for qt in query_tokenized:
        if qt in int_nums:
            values_dict[VALUE_NUM_SYMBOL.split('}')[0]+str(now_idx)] = qt
            query_tmp.append(VALUE_NUM_SYMBOL.split('}')[0]+str(now_idx))
            now_idx += 1
        else:
            query_tmp.append(qt)
    query_tokenized = query_tmp
    query_only_valued = " ".join(query_tokenized)
    nums = float_nums + int_nums

    values = str_1 + str_2
    values_tmp = []
    for item in values:
        values_tmp.append(re.sub('"','',re.sub('\'','',item)))
    values = values_tmp
    
    for idx,val in enumerate(values):
        query_only_valued = query_only_valued.replace(val.strip(), VALUE_STR_SYMBOL.split('}')[0]+str(idx))
        values_dict[VALUE_STR_SYMBOL.split('}')[0]+str(idx)] = val.strip()
        
    #query_tokenized = query.split(' ')
    cols_dict = {}
    for token in query_tokenized:
        if len(token.strip()) == 0:  # in case there are more than one space used
            continue
        if IGNORE_COMMAS_AND_ROUND_BRACKETS:
            keywords_dict = SQL_KEYWORDS_AND_OPERATORS_WITHOUT_COMMAS_AND_BRACES
        else:
            keywords_dict = SQL_KEYWORDS_AND_OPERATORS
        if token.upper() not in keywords_dict and token.split('_')[0] != "{VALUE":
            token = token.upper()
            if USE_COLUMN_AND_VALUE_REPLACEMENT_TOKEN:
                token = re.sub("[T]\d+\.", '', token)
                token = re.sub(r"\"|\'", '', token)
                token = re.sub("[T]\d+", '', token).lower()
#                 if token in table_names_original:
#                     query_keywords.append(TABLE_SYMBOL)
#                     continue
                if token != '' and token in column_names_original:
                    try:
                        tok_ind = column_names_original.index(token)
                    except:
                        print("\ntable: {}".format(table['db_id']))
                        print("\ntoken: {}".format(token))
                        print("column_names_original: {}".format(column_names_original))
                        print("query: {}".format(query))
                        print("query_tokenized: {}".format(query_tokenized))
                        exit(1)
                    col_type = column_types[tok_ind]
                    col_name = column_names[tok_ind]
                    columns.append(col_name)
                    columns.append(token)
                    if token not in cols_dict:
                        cols_dict[token] = COLUMN_SYMBOL.replace("}", str(len(cols_dict)))
                    query_keywords.append(cols_dict[token])
                elif token in table_names_original:
                    query_keywords.append(TABLE_SYMBOL)
                    continue
                    
        else:
            query_keywords.append(token.upper())
    if "FROM" in query_keywords:
        query_keywords = remove_all_from_clauses(query_keywords)

    if USE_LIMITED_KEYWORD_SET:
        query_keywords = [kw for kw in query_keywords if kw in LIMITED_KEYWORD_SET]
    print("columns:{}".format(columns))
    columns_lemed = [lmtzr.lemmatize(w) for w in " ".join(columns).split(" ") if w not in LOW_CHAR]
    columns_lemed_stemed = [ps.stem(w) for w in columns_lemed]
    print("columns_lemed:{}".format(columns_lemed))
    print("columns_lemed_stemed:{}".format(columns_lemed_stemed))
    return " ".join(query_keywords), values, nums, columns_lemed_stemed, cols_dict,values_dict,query_only_valued


def filter_string(cs):
    return "".join([c.upper() for c in cs if c.isalpha() or c == ' '])


def process_question(question, values, nums, columns, values_dict):
    question = " ".join(question).lower()
    values = [re.sub(r"\"|\'", '', val) for val in values]
    for val in values:
        if val in list(values_dict.values()):
            
            print("val: {}".format(val))
            symbol = list (values_dict.keys()) [list (values_dict.values()).index (val)]
            val = val.lower()
            question = re.sub(r'\b'+val+r'\b', symbol, question)
            question = re.sub(r'\b'+re.sub("\'",'',val)+r'\b', symbol, question)
            question = re.sub(r'\b'+re.sub('"','',val)+r'\b', symbol, question)
        elif '\'' + val + '\'' in list(values_dict.values()):
            
            symbol = list (values_dict.keys()) [list (values_dict.values()).index ('\'' + val + '\'')]
            val = val.lower()
            question = re.sub(r'\b'+val+r'\b', symbol, question)
            question = re.sub(r'\b'+re.sub("\'",'',val)+r'\b', symbol, question)
            question = re.sub(r'\b'+re.sub('"','',val)+r'\b', symbol, question)
        elif '"' + val + '"' in list(values_dict.values()):
            
            symbol = list (values_dict.keys()) [list (values_dict.values()).index ('"' + val + '"')]
            val = val.lower()
            question = re.sub(r'\b'+val+r'\b', symbol, question)
            question = re.sub(r'\b'+re.sub("\'",'',val)+r'\b', symbol, question)
            question = re.sub(r'\b'+re.sub('"','',val)+r'\b', symbol, question)
        else:
            continue

    for num in nums:
        num = num.strip()
        try:
            symbol = list (values_dict.keys()) [list (values_dict.values()).index (num)]
            question = re.sub(r'\b'+num+r'\b', symbol, question)
        except:
            continue
    only_valued_question = question

    question_toks = question.split(" ")
    question_lemed = [lmtzr.lemmatize(w) for w in question_toks]
    question_lemed_stemed = [ps.stem(w) for w in question_lemed]
    #replace_inds = [i for i, qt in enumerate(question_lemed_stemed) if qt in columns]
    #print("question_stemed: {}".format(question_stemed))
    #print("replace_inds: {}".format(replace_inds))
    '''
    for ind in replace_inds:
        question_toks[ind] = COLUMN_SYMBOL
    '''
    question_template = ' '.join(question_toks)

    return question_template,only_valued_question

KEY_KEYWORD_SET = {"SELECT", "WHERE", "GROUP", "HAVING", "ORDER", "BY", "LIMIT", "EXCEPT", "UNION", "INTERSECT"}
ALL_KEYWORD_SET = {"SELECT", "WHERE", "GROUP", "HAVING", "DESC", "ORDER", "BY", "LIMIT", "EXCEPT", "UNION", 
                   "INTERSECT", "NOT", "IN", "OR", "LIKE", "(", ">", ")", "COUNT"}

WHERE_OPS = ['=', '>', '<', '>=', '<=', '!=', 'LIKE', 'IS', 'EXISTS']
AGG_OPS = ['MAX', 'MIN', 'SUM', 'AVG']
DASC = ['ASC', 'DESC']

def general_pattern(pattern):
    general_pattern_list = []
    for x in pattern.split(" "):
        if x in KEY_KEYWORD_SET:
            general_pattern_list.append(x)
    
    return " ".join(general_pattern_list)

def sub_pattern(pattern):
    general_pattern_list = []
    for x in pattern.split(" "):
        if x in ALL_KEYWORD_SET:
            general_pattern_list.append(x)
    
    return " ".join(general_pattern_list)

def tune_pattern(pattern):
    general_pattern_list = []
    cols_dict = {}
    for x in pattern.split(" "):
        if "{COLUMN" in x:
            if x not in cols_dict:
                cols_dict[x] = COLUMN_SYMBOL.replace("}", str(len(cols_dict))+"}")
            general_pattern_list.append(cols_dict[x])
            continue
            
        if "{VALUE" in x:
            general_pattern_list.append("{VALUE}")
            continue
            
        if x == 'DISTINCT':
            continue
        elif x in DASC:
            general_pattern_list.append("{DASC}")
        elif x in WHERE_OPS:
            general_pattern_list.append("{OP}")
        elif x in AGG_OPS:
            general_pattern_list.append("{AGG}")
        else:
            general_pattern_list.append(x)
    
    return " ".join(general_pattern_list)


def show_colomns(file_path,table_name,colomn_name):
    if table_name is None:
        print("table_name is None")
        return -1

    with open(file_path,'r') as f:
        conn = sqlite3.connect(f.name)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        tables = [table[0] for table in tables]
        print("conn_table: {}".format(tables))
        df = pd.read_sql_query(f"SELECT {colomn_name} FROM {table_name}",conn)
        conn.close()
    return df


def is_number(s):
    try:  
        float(s)
        return True
    except ValueError:  
        pass  
    try:
        import unicodedata 
        unicodedata.numeric(s)  
        return True
    except (TypeError, ValueError):
        pass
    return False
    
def extract_table_names_more_accurately(sql_query):
    # Updated regex to more accurately extract table names and avoid column names
    regex = r"\bFROM\s+([\w]+)|\bJOIN\s+([\w]+)"

    # Finding all matches using the regex
    matches = re.findall(regex, sql_query, re.IGNORECASE)

    # Extracting table names
    table_names = set()
    for match in matches:
        # match is a tuple, but only one group will be non-None
        table_name = match[0] or match[1]
        if table_name.isidentifier():  # Check if the table name is a valid identifier
            table_names.add(table_name)

    return table_names


def find_value_in_specific_tables(file_path, search_value, known_tables):
    # Connect to the SQLite database
    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()

    # Prepare to search each specified table
    found_locations = []

    for table in known_tables:
        # Query to get the column names of the table
        cursor.execute(f"PRAGMA table_info({table})")
        columns_info = cursor.fetchall()
        columns = [column[1] for column in columns_info]  # column[1] contains the name of the column

        # Check each column for the value
        for column in columns:
            try:
                # Construct the query to search for the value
                query = f"SELECT EXISTS(SELECT 1 FROM {table} WHERE {column}=? LIMIT 1)"
                cursor.execute(query, (search_value,))
                exists = cursor.fetchone()[0]

                # If the value is found, add the table and column to the list
                if exists:
                    found_locations.append((table, column))
            except sqlite3.Error as e:
                # This catch is to handle any SQL errors
                print(f"Error searching in {table}.{column}: {e}")

    conn.close()
    return found_locations

def exchange_value_from_colomns(colomns_in_values,values_dict,question,query,dataset_path,former_values_dict):
    output_data = []
    for key in values_dict.keys():
        #{'Carole': [('Customers', 'first_name')], ('Bernhard', [('Customers', 'last_name')])}
        if len(colomns_in_values[values_dict[key]]) == 0:
            try:
                values_to_change = [math.ceil(float(values_dict[key])*1.1),math.floor(float(values_dict[key])*0.9)]
            except:
                values_to_change = []
        else:
            values_to_change = []
            for table_name,column_name in colomns_in_values[values_dict[key]]:
                with open(dataset_path,'r') as f:
                    conn = sqlite3.connect(f.name)
                    cursor = conn.cursor()
                    df = pd.read_sql_query(f"SELECT {column_name} FROM {table_name}",conn)
                    conn.close()
                exchange_datas = [item[0] for item in df.values.tolist()]
                for item in exchange_datas:
                    if item != values_dict[key] and str(item) != values_dict[key]:
                        values_to_change.append(item)
        values_to_change = list(set(values_to_change))
        if len(values_to_change) == 0:
            continue
        else:
            question_tmp = question
            query_tmp = query
            for other_key in former_values_dict.keys():
                if other_key != key:
                    question_tmp = re.sub(other_key,former_values_dict[other_key],question)
                    query_tmp = re.sub(other_key,former_values_dict[other_key],query)
            for ex_value in values_to_change:
                output_question = re.sub(key,str(ex_value),question_tmp)
                output_query = re.sub(key,str(ex_value),query_tmp)
                output_data.append((output_question,output_query))
    random.shuffle(output_data)
    return output_data


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

def sample_values(val_list):
    #sample 3 if got more than 3,else sample all
    try:
        try:
            values_to_change = random.sample(val_list,3)
        except:
            values_to_change = random.sample(val_list,2)
    except:
        values_to_change = random.sample(val_list,1)

    return values_to_change



def main():

    timeout = 10
    augment_limit = 20

    data_path = '../data/spider_task_stream'
    output_path = '../data/masked_data'
    dataset_path = '../data/spider/database'

    data_paths = []
    output_paths = []
    for i in range(10):
        data_paths.append(os.path.join(data_path,'task_'+str(i)))
        output_paths.append(os.path.join(output_path,'task_'+str(i)))

    for data_in,data_out in zip(data_paths,output_paths):

        training_data, tables = read_in_all_data(data_in)

        train_qq_pairs = get_all_question_query_pairs(training_data)

        print("Training question-query pair count: {}".format(len(train_qq_pairs)))

        pattern_question_dict = defaultdict(list)

        output_all = []

        for eid, (question, query, bd_id) in enumerate(train_qq_pairs):
            table = tables[bd_id]
            if eid % 500 == 0:
                print("processing eid: ", eid)
            
            pattern, values, nums, columns, cols_dict, values_dict,query_only_valued = strip_query(query, table)
            question_template,only_valued_question = process_question(question, values, nums, columns, values_dict)
            
            gen_pattern = general_pattern(pattern)
            more_pattern = sub_pattern(pattern)
            tu_pattern = tune_pattern(pattern)
            
            pattern_question_dict[tu_pattern].append(' '.join(question) + " ||| " + 
                                                    question_template + " ||| " + more_pattern
                                                    + " ||| " + query)
            print("\n--------------------------------------")
            print("original question: {}".format(' '.join(question).encode('utf-8')))
            print("question: {}".format(question_template.encode('utf-8')))
            print("cols_dict: {}".format(cols_dict))
            print("query: {}".format(query))
            print("query_only_valued: {}".format(query_only_valued))
            print("pattern: {}".format(pattern))
            print("values: {}".format(values))
            print("values_dict: {}".format(values_dict))
            print("nums: {}".format(nums))
            print("columns: {}".format(columns))
            print("gen_pattern: {}".format(gen_pattern))
            print("more_pattern: {}".format(more_pattern))
            print("tu_pattern: {}".format(tu_pattern))
            print("table col names: {}".format(table["column_names"]))
            print("query_only_valued{}".format(query_only_valued))

            if 'the number of routes that' in ' '.join(question):
                continue

            former_values_dict = values_dict
            new_values_dict = {}
            for VALUE in values_dict.keys():
                if VALUE in question_template:
                    new_values_dict[VALUE] = values_dict[VALUE]

            values_dict = new_values_dict
            colomns_in_values = {}
            tables_in_query = list(set(extract_table_names_more_accurately(query)))
            print("tables_in_query: {}".format(tables_in_query))
            file_path = os.path.join(dataset_path,bd_id,bd_id+'.sqlite')
            for value_key in values_dict.keys():
                colomns_now = find_value_in_specific_tables(file_path, values_dict[value_key], tables_in_query)
                colomns_in_values[values_dict[value_key]] = colomns_now
            print("colomns_in_values: {}".format(colomns_in_values))
            output_data = exchange_value_from_colomns(colomns_in_values,values_dict,question_template,query_only_valued,file_path,former_values_dict)

            output_tmp = []
            augment_tiaoshu = 0
            for NLQ,SQL in output_data:
                try:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(execute_SQL, SQL, file_path)
                        try:
                            result = future.result(timeout=timeout)
                        except concurrent.futures.TimeoutError:
                            print("time out")
                            result = False
                    #if execute_SQL(SQL,file_path):
                        if result:
                            output_tmp.append((NLQ,SQL))
                            augment_tiaoshu += 1
                            if augment_tiaoshu >= augment_limit:
                                break
                except:
                    continue
            if len(output_tmp) > 0:
                output_data = sample_values(output_tmp)
            else:
                output_data = []

            output_all.append({"question":question,"query":query,"output":output_data})
            
        if not os.path.exists(data_out):
            os.makedirs(data_out)
        dataf_out = os.path.join(data_out,'output.json')
        with open(dataf_out,'w',encoding='utf-8') as f:
            json.dump(output_all, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()

    
