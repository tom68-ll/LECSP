import os
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

from embedding_k_means import k_means
from data_to_embedding import data_embedding

def document_read(data_dir):
    with open(data_dir,'r') as f:
        data = json.load(f)
    return data

def convert_np_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def data_to_cluster(data_dir, output_dir,cluster_dir, model, tokenizer, device,n_clusters):
    embeddings = []
    datas = document_read(data_dir)

    batch_size = 2 #you can change batch size here
    for i in tqdm(range(0, len(datas), batch_size)):
        data_batch = [data["question_masked"] + "|||" + data["query_masked"] for data in datas[i:i + batch_size]]
        batch_embeddings = data_embedding(data_batch, model, tokenizer, device)
        embeddings.extend(batch_embeddings)
    print(len(datas))
    print(len(embeddings))

    clusters, closest_data = k_means(embeddings, n_clusters)
    data_output = []
    for idx, data in enumerate(datas):
        data["cluster"] = clusters[idx]
        data["cluster_center"] = closest_data[idx]
        data["idx"] = idx
        data_output.append(data)

    cluster_items = []
    for i in range(n_clusters):
        cluster_now = {}
        cluster_now['idx'] = i
        cluster_now['patterns'] = []
        cluster_now['center'] = None 
        for data_all in data_output:
            if data_all['cluster'] == i:
                cluster_now['patterns'].append(data_all["pattern_set"])
                if data_all["idx"] == data_all["cluster_center"]:
                    cluster_now["center"] = data_all["pattern_set"]
        if cluster_now['center'] is None and cluster_now['patterns']:
            cluster_now['center'] = cluster_now['patterns'][0]
        cluster_items.append(cluster_now)
                
    with open(output_dir,'w') as f:
        json.dump(data_output,f,default=convert_np_types)
    with open(cluster_dir,'w') as f:
        json.dump(cluster_items,f,default=convert_np_types)

if __name__ == "__main__":
    data_path = "../data/combine1_aug_llm-id/data_pro"
    doc_name = "task_{}.json"
    model_name = "../model/codet5-base"
    output_path = "../data/combine1_aug_llm-id/cluster_result/data"
    cluster_path = "../data/combine1_aug_llm-id/cluster_result/cluster"

    # loading model and tokenizer
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)

    n_clusters=100

    for i in range(7):
        data_dir = os.path.join(data_path, doc_name.format(str(i)))
        output_dir = os.path.join(output_path,doc_name.format(str(i)))
        cluster_dir = os.path.join(cluster_path,doc_name.format(str(i)))
        data_to_cluster(data_dir,output_dir,cluster_dir, model, tokenizer, device,n_clusters)
    