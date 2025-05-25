import os
import json

path_1 = '../data/masked_data'
path_2 = '../data/spider_task_stream'
output_path = '../data/spider_task_stream_new'

file_1 = []
file_2 = []
file_output = []

for i in range(10):
    file_1.append(os.path.join(path_1,'task_'+str(i),'output.json'))
    file_2.append(os.path.join(path_2,'task_'+str(i),'train.json'))
    file_output.append(os.path.join(output_path,'task_'+str(i)))

for i in range(10):
    file_in_1 = file_1[i]
    file_in_2 = file_2[i]
    file_to_out = file_output[i]

    with open(file_in_1,'r') as f:
        data_1 = json.load(f)
    with open(file_in_2,'r') as f:
        data_2 = json.load(f)
    make_new = []
    for item1,item2 in zip(data_1,data_2):
        assert item1["query"] == item2["query"]
        item2["augment_data"] = item1["output"]
        make_new.append(item2)
    if not os.path.exists(file_to_out):
        os.makedirs(file_to_out)
    file_out = os.path.join(file_to_out,'train.json')
    with open(file_out,'w') as f:
        json.dump(make_new,f)