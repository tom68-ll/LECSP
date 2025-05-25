
# LESCP

This is the code of our AAAI 2025 paper **"Filling Memory Gaps: Enhancing Continual Semantic Parsing via SQL Syntax Variance-Guided LLMs Without Real Data Replay"**. 

---

## Environment

### Main dependencies

- Python 3.8+
- transformers
- vllm
- nltk
- pandas
- tqdm
- sqlparse

You can also install our environment via running this code:
```bash
pip install -r requirements.txt
```

## Prepare

### Model
Put **Mixtral-8x7B-Instruct-v0.1** into the 'model' directory.
You can download the model [here](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1).

Put **codet5-base** into the 'model' directory.
You can download the model [here](https://huggingface.co/Salesforce/codet5-base).


### Data
Put **spider** and **spider_task_stream** into the 'data' directory.
You can get the original Spider dataset [here](https://drive.google.com/uc?export=download&id=1TqleXec_OykOYFREKKtschzY29dUcVAQ).

Put **combine1_perm_1** and **wikisql** into the 'data' directory.
You can get the original Spider dataset [here](https://github.com/salesforce/WikiSQL).

## Running Code

### Make multi-tables consisting of tables from WikiSQL
```bash
cd combine_prompt_make
python prompt_make.py
```

### Run the embedding and cluster code to preprocess
Corresponding to sections "Domain Information Elimination" and "Component Bias Analysis" in the paper:
```bash
cd embedding_cluster
python step1_generate_VALUES.py
python step2_synthesis.py
```

### Synthetic data (Intra-Task Memory)
Corresponding to section "Intra-Task Memory Reinforcement" in the paper:
```bash
cd intra
python fstep1_My_generate_VALUES.py
python fstep2_Make_data_concanated.py
python istep1_paraphrase.py
python istep2_paraphrased_process.py
python istep3_remove_alias.py
```
Attention: when execute istep, you should process every task using --task_id

### Synthetic data (Inter-Task Memory)
Corresponding to section "Inter-Task Memory Completion" in the paper.

Generate data for single table:
```bash
cd inter
python ostep0_inference_makesingle.py --start_task_id 0 --end_task_id 6  --batch_size 32 --inference_type V1 --input_max_length 2400 --order 0
python ostep1_outdomain_process_makesingle.py
python ostep2_outdomain_judge_makesingle.py
python ostep3_judged_process_makesingle.py
```

Generate data for multi table:
```bash
cd inter
python ostep0_inference.py --start_task_id 0 --end_task_id 6 --batch_size 32 --inference_type V1 --input_max_length 2400 --order 0
python ostep1_outdomain_process.py
python ostep2_outdomain_judge.py
python ostep3_judged_process.py
```

### Merge data
At last, running the code to merge all the data generated:
```bash
cd data_merge
python merge.py
```

### Training

```bash
bash run.sh
```
---

## Usage Notes

- Each script is independent. Please check the comments in each file for specific usage and parameter settings.
- It is recommended to set up the required Python environment and dependencies before running any scripts.


## Acknowledgements
We gratefully acknowledge the following works that inspired and contributed to this project:
1. [SSCL-Text2SQL](https://github.com/Bahuia/SSCL-Text2SQL)
2. [C3](https://github.com/Bahuia/C3)
3. [Spider](https://github.com/taoyds/spider)
4. [CodeTask-CL](https://github.com/amazon-science/codetask-cl-pptf)


---
## Citation
```
@inproceedings{liu2025filling,
  title={Filling memory gaps: Enhancing continual semantic parsing via sql syntax variance-guided llms without real data replay},
  author={Liu, Ruiheng and Zhang, Jinyu and Song, Yanqi and Zhang, Yu and Yang, Bailong},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={23},
  pages={24641--24649},
  year={2025}
}
```



