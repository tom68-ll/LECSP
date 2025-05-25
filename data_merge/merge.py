import json
import os
from get_text_info import get_text,get_text_forwiki
from tqdm import tqdm
import sqlite3
import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML

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


def contains_letter(s):
    return any(char.isalpha() for char in s)

def process_inter(inter_path_1,inter_path_2,db_path,name_order,task_id):
    with open(inter_path_1,'r') as fp:
        inter_data = json.load(fp)
    '''
    with open(inter_path_2,'r') as fp:
        inter_data += json.load(fp)
    '''
    data_out = []
    if name_order == 'spider':
        for data in tqdm(inter_data):
            db_id = data['db_id']
            schema_path = os.path.join(db_path,db_id,db_id+".sqlite")
            question = data['question']
            if data["text"] == "":
                get_text1 = get_text(schema_path,question)
                text = get_text1.get_text()
            else:
                text = data["text"] + " | " + data["question"]
            data_out.append({
                "text": text,
                "sql": data['query'],
                "example": {
                    "db_id" : db_id
                },
                "type": "OOD",
                "label": data["label"]
            })
    elif name_order == 'wikisql':
        for data in tqdm(inter_data):
            db_id = data['db_id']
            question = data['question']
            if contains_letter(db_id):
                root_path = '../data/wikisql/tables_made'
                sql_path = os.path.join(root_path, f'task_{task_id}_db', f"{db_id}.sql")
                conn = sqlite3.connect(":memory:")
                cursor = conn.cursor()
                with open(sql_path, 'r') as sql_file:
                    sql_script = sql_file.read()
                cursor.executescript(sql_script)
                conn.commit()
                
                # Pass the open connection to get_text
                get_text1 = get_text(None, question, conn)
                text = get_text1.get_text()
                
                conn.close()
            else:
                get_text1 = get_text_forwiki(db_id,question)
                text = get_text1.get_text()
            data_out.append({
                "text": text,
                "sql": data['query'],
                "example": {
                    "db_id" : db_id
                },
                "type": "OOD",
                "label": data["label"]
            })
    return data_out

def process_intra(intra_path):
    with open(intra_path,'r') as fp:
        intra_data = json.load(fp)
    data_out = []
    for data in tqdm(intra_data):
        text = data['text'].split('|')[0] + '|' + data['question']
        data_out.append({
            "text": text,
            "sql": data['query'],
            "example": {
                "db_id" : data['db_id']
            },
            "type": "ID"
        })
    return data_out

def output(auxiliary_path,inter_data,intra_data,out_path):
    #train
    train_path = os.path.join(auxiliary_path,"train.json")
    with open(train_path,'r') as fp:
        original_train_data = json.load(fp)
    for data in tqdm(original_train_data):
        data["type"] = "ORI"
        replacer = SQLAliasReplacer(data["sql"])
        data["sql"] = replacer.replace_aliases()
        

    output_train_data = original_train_data + intra_data + inter_data
    #output_train_data = original_train_data + intra_data
    print(len(intra_data),len(inter_data))
    #dev
    dev_path = os.path.join(auxiliary_path,"dev.json")
    with open(dev_path,'r') as fp:
        output_dev_data = json.load(fp)
    for data in tqdm(output_dev_data):
        replacer = SQLAliasReplacer(data["sql"])
        data["sql"] = replacer.replace_aliases()
    
    #test
    test_path = os.path.join(auxiliary_path,"test.json")
    with open(test_path,'r') as fp:
        output_test_data = json.load(fp)
    for data in tqdm(output_test_data):
        replacer = SQLAliasReplacer(data["sql"])
        data["sql"] = replacer.replace_aliases()

    out_train_dir = os.path.join(out_path,"train.json")
    out_dev_dir = os.path.join(out_path,"dev.json")
    out_test_dir = os.path.join(out_path,"test.json")

    with open(out_train_dir,'w') as fp:
        json.dump(output_train_data,fp)

    with open(out_dev_dir,'w') as fp:
        json.dump(output_dev_data,fp)

    with open(out_test_dir,'w') as fp:
        json.dump(output_test_data,fp)



if __name__ == '__main__':
    output_path = "../data/only_task0_order0/Mistral/mistral_augment_data"
    auxiliary_path = "../data/combine1_perm_1"
    inter_path_1 = "../data/only_task0_order0/Mistral/step3_out"
    #inter_path_2 = "../data/classification_inter_data_2/step3_out"
    intra_path = "../data/intra_data/step3_out"

    datadase_path = "../inter/prompt_text_to_sql/data_processed/spider-train/database"
    name_orders = ['spider','wikisql','spider','wikisql','spider','wikisql','spider']
    #name_orders = ['wikisql','spider','spider','spider','spider','wikisql','wikisql']
    for task_id in range(1):
        name_order = name_orders[task_id]
        task_out_path = os.path.join(output_path,"task_" + str(task_id))
        if not os.path.exists(task_out_path):
            os.mkdir(task_out_path)
        task_auxiliary_path = os.path.join(auxiliary_path,"task_" + str(task_id))

        inter_dir_1 = os.path.join(inter_path_1,"task_" + str(task_id) + ".json")
        #inter_dir_2 = os.path.join(inter_path_2,"task_" + str(task_id) + ".json")
        intra_dir = os.path.join(intra_path,"task_" + str(task_id) + ".json")
        #intra_data = []
        inter_dir_2 = ""


        inter_data = process_inter(inter_dir_1,inter_dir_2,datadase_path,name_order,task_id)
        intra_data = process_intra(intra_dir)

        output(task_auxiliary_path,inter_data,intra_data,task_out_path)


