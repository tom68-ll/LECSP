import json
import os

root_path = '../data/combine1_perm_1'
tasks_name = ['task_1','task_3','task_5']

table_name_path = '../data/wikisql/tables.json'
table_content_path = '../data/wikisql/table_content.json'

def make_prompt(table_content, table_name) -> str:
    # Replace spaces with underscores in the table name
    table_name = table_name.replace(' ', '_')
    # Replace spaces with underscores in the column names
    columns_info = ", ".join([col.replace(' ', '_') for col in table_content["header"]])
    table_info = f"CREATE TABLE {table_name} ({columns_info});"
    
    insert_infos = []
    for row_info in table_content["rows"]:
        formatted_values = ", ".join(f"'{value}'" for value in row_info)
        insert_statement = f"INSERT INTO {table_name} ({columns_info}) VALUES ({formatted_values});"
        insert_infos.append(insert_statement)
    
    output = f"{table_info}\n" + "\n".join(insert_infos) + "\n\n" + "Please refine the above SQL statement into a standard format, and create two more tables that can be linked to the table above using a foreign key, and add values to each table. The SQL statement that finally returns this database (contains three tables)"
    return output

for task_name in tasks_name:
    prompt_list = []
    db_id_list = {}
    data_path = os.path.join(root_path, task_name, "train.json")
    with open(data_path, 'r') as fp:
        data = json.load(fp)
    for item in data:
        if item['text'].split('|')[0] not in db_id_list.keys():
            db_id_list[item['text'].split('|')[0]] = item["example"]["db_id"]
    with open(table_name_path, 'r') as fp:
        table_name_data = json.load(fp)
    with open(table_content_path, 'r') as fp:
        table_content_data = json.load(fp)
    for key in list(db_id_list.keys()):
        # Find the real name
        table_name = ""
        table_content = {}
        for item in table_name_data:
            if item["db_id"] == db_id_list[key]:
                table_name = item["table_names"][0]
        for content in list(table_content_data.keys()):
            if content == db_id_list[key]:
                table_content = table_content_data[content]
        prompt_list.append(make_prompt(table_content, table_name))
    with open(f'prompt_{task_name}.json', 'w') as fp:
        json.dump(prompt_list, fp,ensure_ascii=False)
