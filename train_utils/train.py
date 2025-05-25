import sys

sys.path.append('.')

from trainer import FinetunePredictorTrainer, get_tasks_results, eval_stream_task

import argparse
import json
import os
import numpy as np

os.environ["WANDB_DISABLED"] = "true"

parser = argparse.ArgumentParser('')
parser.add_argument('--cuda_visible_devices', type=str, default='3')
parser.add_argument('--backbone_plm', type=str, default='t5-large-lm-adapt')
parser.add_argument('--per_device_train_batch_size', type=int, default=3)
parser.add_argument('--per_device_eval_batch_size', type=int, default=3)
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
parser.add_argument('--model_name_or_path', type=str, default='')
parser.add_argument('--total_output_dir', type=str, default='train/ckpt/predictor/finetune')
parser.add_argument('--total_dataset_dir', type=str, default='train/data_train/predictor')
parser.add_argument('--root_output_dir', type=str, default='out')

parser.add_argument('--dataset_type', type=str, default='combine')
parser.add_argument('--dataset', type=str, default='')
parser.add_argument('--task_num', type=int, default=7)
parser.add_argument('--first_task_id', type=int, default=0)
parser.add_argument('--last_task_id', type=int, default=6)
parser.add_argument('--do_cl_eval', type=bool, default=True)
parser.add_argument('--do_wo_stu_ablation', type=bool, default=False)
parser.add_argument('--do_first_seen_eval', type=bool, default=False)
parser.add_argument('--seed', type=int, default=2022)

parser.add_argument('--metric_for_best_model', type=str, default='exact_match')
parser.add_argument('--greater_is_better', type=bool, default=True)
parser.add_argument('--max_source_length', type=int, default=512)
parser.add_argument('--max_target_length', type=int, default=256)
parser.add_argument('--overwrite_output_dir', type=bool, default=False)
parser.add_argument('--resume_from_checkpoint', type=bool, default=None)
parser.add_argument('--do_train', type=bool, default=True)
parser.add_argument('--do_eval', type=bool, default=True)
parser.add_argument('--do_predict', type=bool, default=True)
parser.add_argument('--predict_with_generate', type=bool, default=True)
parser.add_argument('--lr_scheduler_type', type=str, default='linear')
parser.add_argument('--label_smoothing_factor', type=float, default=0.0)
parser.add_argument('--warmup_ratio', type=float, default=0.0)
parser.add_argument('--warmup_steps', type=int, default=0)
parser.add_argument('--num_beams', type=int, default=4)
parser.add_argument('--optim', type=str, default='adafactor')
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--adam_epsilon', type=float, default=1e-06)
parser.add_argument('--load_best_model_at_end', type=bool, default=True)
parser.add_argument('--num_train_epochs', type=int, default=1000)
parser.add_argument('--save_strategy', type=str, default='steps')
parser.add_argument('--save_total_limit', type=int, default=1)
parser.add_argument('--evaluation_strategy', type=str, default='steps')

parser.add_argument('--shuffle_tables', type=bool, default=False)
parser.add_argument('--shuffle_columns', type=bool, default=False)
parser.add_argument('--only_predict', type=bool, default=False)
parser.add_argument('--start_predict_task_id', type=int, default=0)
parser.add_argument('--end_predict_task_id', type=int, default=0)
parser.add_argument('--full_data', type=bool, default=False)
parser.add_argument('--alpha', type=float, default=0.0)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--aug', action='store_true')
parser.add_argument('--fast_debug', type=int, default=0)


args = parser.parse_args()

if args.backbone_plm.startswith('t5-large'):
    dir_name = 'finetune_t5-large'
elif args.backbone_plm.startswith('t5-base'):
    dir_name = 'finetune_t5-base'
elif args.backbone_plm.startswith('t5-small'):
    dir_name = 'finetune_t5-small'
else:
    raise NotImplementedError


args.dataset_dir = f'train/data_train/{args.dataset_type}/{args.dataset}'
# args.output_dir = f'train/ckpt/{args.dataset_type}/{args.dataset}/{dir_name}'
args.output_dir = f'{args.root_output_dir}/{dir_name}'

if __name__ == '__main__':
    predictor_trainer = FinetunePredictorTrainer(args)

    args_dict = vars(args)
    config_pth = f'{args.output_dir}/config.json'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    with open(config_pth, 'w') as file:
        json.dump(args_dict, file)

    if not args.only_predict:
        for task_id in range(args.first_task_id, args.last_task_id + 1):
            # Load different trainers according to task_id
            print("="*10, f"Task_{task_id}", "="*10)
            if task_id == 0:
                predictor_trainer = FinetunePredictorTrainer(args)
            else:
                predictor_trainer= FinetunePredictorTrainer(args)
            predictor_trainer.train(task_id)

    # Test stream results
    print("="*10, "Only_predict", "="*10)
    print(f"Model_dir: {args.backbone_plm}")
    print(f"dataset_dir: {args.dataset}")
    print(f"Checkpoint_dir: {args.root_output_dir}")
    
    # predictor_trainer = FinetunePredictorTrainer(args) 
    # predictor_trainer.predict(1, 1, stream=False)

    for td in range(args.start_predict_task_id, args.end_predict_task_id+1):
        current_task_id = td
        folder_path = os.path.join('train/data_train', f"{args.dataset_type}", f"{args.dataset}")
        file_count = 0
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isdir(file_path):
                file_count += 1
        
        if file_count == 0:
            assert 1 == 2
        last_task_id = file_count - 1

        if current_task_id == last_task_id:
            range_id = last_task_id
        else:
            range_id = current_task_id + 1
        for to_id in range(0, range_id+1):
            print("="*10, f"Model_{current_task_id} tests on Task_{to_id}", "="*10)
            predictor_trainer = FinetunePredictorTrainer(args, task_id=current_task_id)
            predictor_trainer.predict(current_task_id, to_id, stream=True)


    ############### Calculating stream metrics ###############
    folder_path = os.path.join('train/data_train', f"{args.dataset_type}", f"{args.dataset}")
    file_count = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if os.path.isdir(file_path):
            file_count += 1

    if file_count == 0:
        assert 1 == 2
    last_task_id = file_count - 1
    
    result = {}
    test_size_dic, em_acc_array, ex_acc_array = get_tasks_results(args.dataset, 
                                                                  args.backbone_plm, 
                                                                  args.root_output_dir, 
                                                                  args.dataset_type,
                                                                  task_num=(last_task_id+1))
                                                                  
    print(test_size_dic)
    print(em_acc_array)
    print(ex_acc_array)
    result['test_size'] = test_size_dic
    result['EM_array'] = em_acc_array.tolist()
    result['EX_array'] = ex_acc_array.tolist()

    stream_result = eval_stream_task(task_num=(last_task_id+1), test_num=test_size_dic, em_acc=em_acc_array, ex_acc=ex_acc_array)
    metirc = {}
    for key, value in stream_result.items():
        print((f"{key}: {round(value[-1], 4)}"))
        metirc[key] = round(value[-1], 4)
    result['metric'] = metirc

    # save
    path = os.path.join(args.output_dir, 'metrics.json')
    print(f"Save result in {path}")
    with open(path, 'w') as f:
        json.dump(result, f) 
    print(f"[INFO] task num: {last_task_id+1}")






    