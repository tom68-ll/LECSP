import os
import re
import json
from template_config import *
import nltk
from collections import defaultdict
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from tqdm import tqdm
import sqlite3
import pandas as pd
import re

nltk.download('wordnet')

ps = PorterStemmer()
lmtzr = WordNetLemmatizer()

KEY_KEYWORD_SET = {"SELECT", "WHERE", "GROUP", "HAVING", "ORDER", "BY", "LIMIT", "EXCEPT", "UNION", "INTERSECT"}
ALL_KEYWORD_SET = {"SELECT", "WHERE", "GROUP", "HAVING", "DESC","ASC", "ORDER", "BY", "LIMIT", "EXCEPT", "UNION", 
                   "INTERSECT", "NOT", "IN", "OR", "LIKE", "(", ">", ")", "COUNT"}

WHERE_OPS = ['=', '>', '<', '>=', '<=', '!=', 'LIKE', 'IS', 'EXISTS']
AGG_OPS = ['MAX', 'MIN', 'SUM', 'AVG']
DASC = ['ASC', 'DESC']

dql_keywords = {
    "SELECT", "FROM", "WHERE", "GROUP", "BY", "HAVING", 
    "ORDER", "LIMIT", "OFFSET", "DISTINCT", "FROM", "JOIN"
    "AND", "OR", "NOT", "IN", 
    "LIKE", "ILIKE", "BETWEEN", "INTERSECT", "EXCEPT", 
    "UNION", "(", ")", "{OP}", "{DASC}",
    "AVG", "COUNT", "SUM", "MIN", "MAX"
}



def read_in_all_data(data_path=DATA_PATH,name_order='spider'):
    if name_order == 'spider':
        tables_path = "../data/spider_task_stream"
        training_data = json.load(open(os.path.join(data_path, "train.json")))

        tables = {}
        for task_id in range(10):
            tables_org = json.load(open(os.path.join(tables_path, "task_" + str(task_id), "tables.json")))
            for tab in tables_org:
                tables[tab['db_id']] = tab
        
        train_data_out = []
        for item in training_data:
            train_data_out.append(item)
    elif name_order == 'wikisql':
        tables_path = '../data/wikisql'
        training_data = json.load(open(os.path.join(data_path, "train.json")))

        tables = {}
        tables_org = json.load(open(os.path.join(tables_path, "tables.json")))
        for tab in tables_org:
            tables[tab['db_id']] = tab
        
        train_data_out = []
        for item in training_data:
            train_data_out.append(item)


    return train_data_out, tables

def get_all_question_query_pairs(data):
    question_query_pairs = []
    for item in data:
        #question_query_pairs.append((item['question_toks'], item['query'], item['db_id']))
        question_query_pairs.append((item['example']['question_toks'], item['sql'], item['example']['db_id']))
    return question_query_pairs



def remove_all_from_clauses(query_keywords):
    """
    remove all keywords from from clauses, until there is no more from clauses
    e.g. select {} from {} as {} where {} = {} --> select {} where {} = {}
    """
    # remove from clause by deleting the range from "FROM" to "WHERE" or "GROUP"
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
                end_token_locations.append(query_keywords.index(end_token, from_location))
            except ValueError:
                pass

        query_keywords = query_keywords[:from_location] + [FROM_SYMBOL] + query_keywords[min(end_token_locations):]
        
    return query_keywords


def strip_query(query, table,name_order):
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
                    if name_order == 'spider':
                        col_type = column_types[tok_ind]
                        col_name = column_names[tok_ind]
                    elif name_order == 'wikisql':
                        col_type = column_types[tok_ind-1]
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
    query_keywords_withfrom = query_keywords
    if "FROM" in query_keywords:
        query_keywords = remove_all_from_clauses(query_keywords)

    if USE_LIMITED_KEYWORD_SET:
        query_keywords = [kw for kw in query_keywords if kw in LIMITED_KEYWORD_SET]
        query_keywords_withfrom = [kw for kw in query_keywords_withfrom if kw in LIMITED_KEYWORD_SET]
    columns_lemed = [lmtzr.lemmatize(w) for w in " ".join(columns).split(" ") if w not in LOW_CHAR]
    columns_lemed_stemed = [ps.stem(w) for w in columns_lemed]
    return " ".join(query_keywords)," ".join(query_keywords_withfrom), values, nums, columns_lemed_stemed, cols_dict,values_dict,query_only_valued


def process_question(question, values, nums, columns, values_dict):
    question = " ".join(question).lower()
    values = [re.sub(r"\"|\'", '', val) for val in values]
    for val in values:
        if val in list(values_dict.values()):
            
            symbol = list (values_dict.keys()) [list (values_dict.values()).index (val)]
            val = val.lower()
            escaped_val = re.escape(val)
            question = re.sub(r'\b' + escaped_val + r'\b', symbol, question)
            question = re.sub(r'\b' + re.sub("'", '', escaped_val) + r'\b', symbol, question)
            question = re.sub(r'\b' + re.sub('"', '', escaped_val) + r'\b', symbol, question)
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
    replace_inds = [i for i, qt in enumerate(question_lemed_stemed) if qt in columns]
    #print("question_stemed: {}".format(question_stemed))
    #print("replace_inds: {}".format(replace_inds))
    
    for ind in replace_inds:
        question_toks[ind] = COLUMN_SYMBOL
    
    question_template = ' '.join(question_toks)

    return question_template,only_valued_question

def general_pattern(pattern):
    general_pattern_list = []
    for x in pattern.split(" "):
        if x in KEY_KEYWORD_SET:
            general_pattern_list.append(x)
    
    return " ".join(general_pattern_list)


def find_nested_parentheses_content(text):
    stack = []
    result = []
    temp = ""

    for char in text:
        if char == '(':

            if stack:
                temp += char
            stack.append(char)
        elif char == ')' and stack:

            stack.pop()

            if stack:
                temp += char
            else:

                result.append(temp)
                temp = ""
        elif stack:

            temp += char

    return result

def sub_pattern(pattern):
    general_pattern_list = []
    for x in pattern.split(" "):
        if x in dql_keywords:
            general_pattern_list.append(x)
    tmp_pattern = " ".join(general_pattern_list)

    contents = find_nested_parentheses_content(tmp_pattern)
    for content in contents:
        if content in tmp_pattern:
            #mask the nesting
            if "SELECT" in content:
                tmp_pattern = tmp_pattern.replace(content, "NESTING")
            
    return tmp_pattern

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
            
        if x in DASC:
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

#should be val_col_dict
def find_col_table_and_value_for_where(simple_query,col_table_dict,col_val_dict):
    #{'alias':'table'}
    table_alias = {}
    if isinstance(simple_query['from'],list):
        for item in simple_query['from']:
            if 'join' in item.keys():
                table_alias[item['join']['name']] = item['join']['value']
            else:
                 table_alias[item['name']] = item['value']
    
    def process_where(item,ncol_table_dict,ncol_val_dict):
        for key in item.keys():
            if isinstance(item[key][1],dict):
                    value_key = list(item[key][1].keys())[0]
                    value_now = item[key][1][value_key]
            elif len(item[key]) > 2:
                value_now = []
                for value_two in item[key][1:]:
                    value_now.append(value_two)
            else:
                    value_now = item[key][1]
            if len(table_alias.keys()) == 0:
                col_name = item[key][0]
                ncol_table_dict[col_name] = simple_query['from']
            elif len(item[key][0].split('.')) == 2:
                col_name = item[key][0].split('.')[-1]
                ncol_table_dict[col_name] = table_alias[item[key][0].split('.')[0]]
            elif len(item[key][0].split('.')) == 1:
                col_name = item[key][0]#dataset_path
                for table_name_keys in table_alias.keys():
                    table_name_value = table_alias[table_name_keys]
                    try:
                        show_colomns(dataset_path,table_name_value,col_name)
                        ncol_table_dict[col_name] = table_name_value
                        break
                    except:
                        continue
            
            if isinstance(value_now,list):
                for val_now in value_now:
                    ncol_val_dict[val_now] = col_name
            
            #where': {'nin': ['station_id', {'select': {'value': 'id'}, 'from': 'station', 'where': {'eq': ['city', 'Palo Alto']}}]}}
            elif isinstance(value_now,dict):
                ncol_table_dict_tmp,ncol_val_dict_tmp = find_col_table_and_value(item[key][1])
                ncol_table_dict.update(ncol_table_dict_tmp)
                ncol_val_dict.update(ncol_val_dict_tmp)
            else:
                ncol_val_dict[value_now] = col_name
        return ncol_table_dict,ncol_val_dict

    def process_and_or(where_value,bcol_table_dict,bcol_val_dict):
        if 'and' in where_value.keys():
            for item in where_value['and']:
                bcol_table_dict,bcol_val_dict = process_and_or(item,bcol_table_dict,bcol_val_dict)
        elif 'or' in where_value.keys():
            for item in where_value['or']:
                bcol_table_dict,bcol_val_dict = process_and_or(item,bcol_table_dict,bcol_val_dict)
        else:
            bcol_table_dict,bcol_val_dict = process_where(where_value,bcol_table_dict,bcol_val_dict)
        return bcol_table_dict,bcol_val_dict
    col_table_dict,col_val_dict = process_and_or(simple_query['where'],col_table_dict,col_val_dict)
    '''
    if 'and' in simple_query['where'].keys():
        for item in simple_query['where']['and']:
            col_table_dict,col_val_dict = process_where(item,col_table_dict,col_val_dict)
    elif 'or' in simple_query['where'].keys():
        for item in simple_query['where']['or']:
            col_table_dict,col_val_dict = process_where(item,col_table_dict,col_val_dict)
    else:
        col_table_dict,col_val_dict = process_where(simple_query['where'],col_table_dict,col_val_dict)
    '''
    return col_table_dict,col_val_dict

def find_col_table_and_value_for_having(simple_query,col_table_dict,col_val_dict):
    table_alias = {}
    if isinstance(simple_query['from'],list):
        for item in simple_query['from']:
            if 'join' in item.keys():
                table_alias[item['join']['name']] = item['join']['value']
            else:
                table_alias[item['name']] = item['value']

    for having_keys in simple_query['having'].keys():
        having_values = simple_query['having'][having_keys]

        if isinstance(having_values[1],dict):
            col_table_dict,col_val_dict = find_col_table_and_value(having_values[1])
        elif isinstance(having_values[0],dict):
            if list(having_values[0].keys())[0] == 'count':
                col_val_dict[having_values[1]] = 'To_Count'
            elif list(having_values[0].keys())[0] == 'avg':
                col_val_dict[having_values[1]] = 'To_Avg'
            elif list(having_values[0].keys())[0] == 'sum':
                col_val_dict[having_values[1]] = 'To_Sum'
            elif list(having_values[0].keys())[0] == 'max':
                col_val_dict[having_values[1]] = 'To_Max'
            elif list(having_values[0].keys())[0] == 'min':
                col_val_dict[having_values[1]] = 'To_Min'
            else:
                return False
        else:
            return False
    
    return col_table_dict,col_val_dict

def find_col_table_and_value_for_limit(simple_query,col_table_dict,col_val_dict):
    table_alias = {}
    if isinstance(simple_query['from'],list):
        for item in simple_query['from']:
            if 'join' in item.keys():
                table_alias[item['join']['name']] = item['join']['value']
            else:
                table_alias[item['name']] = item['value']
    
    col_val_dict[simple_query['limit']] = 'To_Limit'
    
    return col_table_dict,col_val_dict

def find_col_table_and_value(parsed_query):
    col_table_dict = {}
    col_val_dict = {}
    if 'intersect' in parsed_query.keys():
        for values in parsed_query['intersect']:
            if 'where' in values.keys():
                col_table_dict,col_val_dict = find_col_table_and_value_for_where(values,col_table_dict,col_val_dict)
            if 'having' in values.keys():
                col_table_dict,col_val_dict = find_col_table_and_value_for_having(values,col_table_dict,col_val_dict)
            if 'limit' in values.keys():
                col_table_dict,col_val_dict = find_col_table_and_value_for_limit(values,col_table_dict,col_val_dict)
    elif 'union' in parsed_query.keys():
        for values in parsed_query['union']:
            if 'where' in values.keys():
                col_table_dict,col_val_dict = find_col_table_and_value_for_where(values,col_table_dict,col_val_dict)
            if 'having' in values.keys():
                col_table_dict,col_val_dict = find_col_table_and_value_for_having(values,col_table_dict,col_val_dict)
            if 'limit' in values.keys():
                col_table_dict,col_val_dict = find_col_table_and_value_for_limit(values,col_table_dict,col_val_dict)
    elif 'except' in parsed_query.keys():
        for values in parsed_query['except']:
            if 'where' in values.keys():
                col_table_dict,col_val_dict = find_col_table_and_value_for_where(values,col_table_dict,col_val_dict)
            if 'having' in values.keys():
                col_table_dict,col_val_dict = find_col_table_and_value_for_having(values,col_table_dict,col_val_dict)
            if 'limit' in values.keys():
                col_table_dict,col_val_dict = find_col_table_and_value_for_limit(values,col_table_dict,col_val_dict)
    else:
        if 'where' in parsed_query.keys():
            col_table_dict,col_val_dict = find_col_table_and_value_for_where(parsed_query,col_table_dict,col_val_dict)
        if 'having' in parsed_query.keys():
            col_table_dict,col_val_dict = find_col_table_and_value_for_having(parsed_query,col_table_dict,col_val_dict)
        if 'limit' in parsed_query.keys():
                col_table_dict,col_val_dict = find_col_table_and_value_for_limit(parsed_query,col_table_dict,col_val_dict)
    return col_table_dict,col_val_dict

def JOIN_process(pattern_masked,pattern_with_from):
    FROM_situration = []
    from_number = -1
    for token in pattern_with_from.split():
        if token == "FROM":
            from_number += 1
            FROM_situration.append(False)
        elif token == "JOIN":
            FROM_situration[from_number] = True
        else:
            continue
    
    from_number = 0
    new_masked_pattern = []
    for idx,token in enumerate(pattern_masked):
        new_masked_pattern.append(token)
        if token == "{FROM}":
            if FROM_situration[from_number] is True:
                new_masked_pattern.append("{JOIN}")
            from_number += 1
    return new_masked_pattern
dataset_name = 'combine' # 'combine'
if dataset_name == 'combine':   
    names_order = ['spider','wikisql','spider','wikisql','spider','wikisql','spider']
elif dataset_name == 'wikisql':
    names_order = ['wikisql', 'wikisql', 'wikisql', 'wikisql', 'wikisql', 'wikisql', 'wikisql', 'wikisql', 'wikisql', 'wikisql']
task_num = 7
#task_num = 10
for task_id in range(task_num):
    name_order = names_order[task_id]
    training_data, tables = read_in_all_data("../data/combine1_perm_1/task_" + str(task_id),names_order[task_id])
    help_to_save = '../data/combine1_aug_llm-id/data_pro/task_'+str(task_id) + '.json'
    train_qq_pairs = get_all_question_query_pairs(training_data)
    print("Training question-query pair count: {}".format(len(train_qq_pairs)))

    timeout = 10
    dataset_path = '../data/spider/database'
    pattern_question_dict = defaultdict(list)
    output_all = []

    for eid, (question, query, bd_id) in tqdm(enumerate(train_qq_pairs),total=len(train_qq_pairs)):
        table = tables[bd_id]
        if eid % 500 == 0:
            print("processing eid: ", eid)
        
        pattern,pattern_with_from, values, nums, columns, cols_dict, values_dict,query_only_valued = strip_query(query, table,name_order)
        question_template,only_valued_question = process_question(question, values, nums, columns, values_dict)
        
        gen_pattern = general_pattern(pattern)
        tu_pattern = tune_pattern(pattern_with_from)
        more_pattern = sub_pattern(tu_pattern)
        
        
        pattern_question_dict[tu_pattern].append(' '.join(question) + " ||| " + 
                                                question_template + " ||| " + more_pattern
                                                + " ||| " + query)
        
        pattern_set = more_pattern.split()
        pattern_tmp = []
        for patte in pattern_set:
            if patte == '(' or patte == ')':
                continue
            elif patte == "ORDER":
                pattern_tmp.append("ORDER BY")
            elif patte == "GROUP":
                pattern_tmp.append("GROUP BY")
            elif patte == "BY":
                continue
            elif patte == "(NESTING)":
                pattern_tmp.append("{NESTING}")
            else:
                pattern_tmp.append(patte)
        pattern_set = pattern_tmp
        
        
        tmp_pattern = pattern.split()
        
        pattern_masked = []
        for i, elem in enumerate(tmp_pattern):
            if elem.startswith("{COLUMN"):
                pattern_masked.append("{COLUMN}")
            elif elem.startswith("{VALUE"):
                pattern_masked.append(elem+'}')
            else:
                pattern_masked.append(elem)
        #process join
        pattern_masked = JOIN_process(pattern_masked,pattern_with_from)
        pattern_masked = " ".join(pattern_masked)

        #remove sames
        pattern_tmp = []
        for item in pattern_set:
            if pattern_tmp == []:
                pattern_tmp.append(item)
                continue
            if item == pattern_tmp[-1]:
                continue
            pattern_tmp.append(item)
        pattern_set = pattern_tmp

        if "{JOIN}" in pattern_masked:
            pattern_tmp = []
            for item in pattern_set:
                pattern_tmp.append(item)
                if item == 'FROM':
                    pattern_tmp.append('JOIN')
            pattern_set = pattern_tmp

        question_template = question_template.split()
        question_masked = []
        for i, elem in enumerate(question_template):
            if elem.startswith("{COLUMN"):
                question_masked.append("{COLUMN}")
            elif elem.startswith("{VALUE"):
                question_masked.append(elem+'}')
            else:
                question_masked.append(elem)
        question_masked = " ".join(question_masked)

        output_data = {}
        output_data["db_id"] = bd_id
        output_data['question'] = question
        output_data['query'] = query
        output_data['question_masked'] = question_masked
        output_data['query_masked'] = pattern_masked
        output_data['pattern_set'] = list(pattern_set)
        output_all.append(output_data)

        

    with open(help_to_save,'w') as f:
        json.dump(output_all,f)

