import json
import os

data_in = "../data/spider_perm_1"
data_out = "./data"

name = "test.json"

for task_id in range(11):
    task_in = os.path.join(data_in,"task_" + str(task_id),name)
    task_out = os.path.join(data_out,"task_" + str(task_id),name)
    with open(task_in,'r') as fp:
        data = json.load(fp)
    with open(task_out,'w') as fp:
        json.dump(data,fp) 