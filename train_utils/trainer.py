import json
import logging
import os
import shutil
import sys
import datasets
import nltk
import torch
from torch.utils.data import Dataset, DataLoader
import math
import time
import random
import transformers
from filelock import FileLock
from typing import Optional, List, NamedTuple
from dataclasses import dataclass, field
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import copy

from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    EarlyStoppingCallback,
)
from transformers import AdamW, Trainer
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import speed_metrics, get_last_checkpoint
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers import Adafactor, set_seed, T5ForConditionalGeneration, AutoTokenizer
from predictor_base import PredictorTrainer
from spider.evaluation import evaluate as spider_evaluate
from wikisql.evaluation import evaluate as wikisql_evaluate
from torch.nn import functional as F
from transformers.trainer_pt_utils import IterableDatasetShard
from transformers.trainer_utils import seed_worker


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                    "the model's position embeddings."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
                    "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                    "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                    "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                    "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the decoder_start_token_id."
                    "Useful for multilingual models like mBART where the first generated token"
                    "needs to be the target language token (Usually it is the target language token)"
        },
    )

    def __post_init__(self):
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    total_output_dir: str = field(
        default=None
    )
    dataset: str = field(
        default=None
    )
    total_dataset_dir: str = field(
        default=None
    )
    dataset_dir: str = field(
        default=None
    )



class EvalPrediction(NamedTuple):
    predictions: List[str]
    gold_sql: List[str]
    gold_db: List[str]


class T5ForConditionalGenerationWithKL(T5ForConditionalGeneration):
    def __init__(self, config, kl_model=None):
        super().__init__(config)
        self.kl_model = kl_model  # 另一个用于计算KL散度的模型

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        domain_type=None,  # 增加一个指示输入是否域外的参数
        **kwargs
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            **kwargs,
        )

        # 计算交叉熵损失
        loss = outputs.loss

        if domain_type == "OOD":
            kl_loss = torch.tensor(0.0).to(loss.device)  # 初始化KL损失
            if self.kl_model is not None:
                # 对于域外样本，计算KL散度
                with torch.no_grad():  # 不更新另一个模型的权重
                    kl_model_outputs = self.kl_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        decoder_attention_mask=decoder_attention_mask,
                        labels=labels,
                    )
                # 计算KL散度
                kl_loss = F.kl_div(
                    F.log_softmax(outputs.logits, dim=-1),
                    F.softmax(kl_model_outputs.logits, dim=-1),
                    reduction='batchmean'
                )

            # 更新总损失
            loss += kl_loss

        return loss  # 返回更新后的损失


class SubTrainer(Seq2SeqTrainer):
    def __init__(self, dataset_type: str, **kwargs):
        super().__init__(**kwargs)
        self.compute_metrics = self._compute_metrics
        self.output_dir = self.args.output_dir
        self.dataset_type = dataset_type
        self.spider_db_ids = ['wine_1', 'game_injury', 'assets_maintenance', 'county_public_safety', 'farm', 'wedding',
                              'architecture', 'machine_repair', 'battle_death', 'debate', 'behavior_monitoring',
                              'perpetrator', 'college_2', 'csu_1', 'hospital_1', 'college_1', 'college_3', 'flight_2',
                              'flight_4', 'ship_mission', 'ship_1', 'flight_1', 'railway', 'aircraft', 'pilot_record',
                              'flight_company', 'train_station', 'restaurants', 'yelp', 'restaurant_1', 'theme_gallery',
                              'cre_Theme_park', 'roller_coaster', 'inn_1', 'apartment_rentals', 'coffee_shop',
                              'museum_visit', 'film_rank', 'imdb', 'program_share', 'entertainment_awards', 'cinema',
                              'tvshow', 'movie_1', 'insurance_policies', 'loan_1', 'small_bank_1', 'solvency_ii',
                              'insurance_fnol', 'insurance_and_eClaims', 'real_estate_properties',
                              'tracking_software_problems', 'network_2', 'allergy_1', 'protein_institute',
                              'dog_kennels', 'network_1', 'medicine_enzyme_interaction', 'device', 'station_weather',
                              'pets_1', 'twitter_1', 'storm_record', 'browser_web', 'wta_1', 'match_season', 'wrestler',
                              'gymnast', 'bike_1', 'body_builder', 'race_track', 'formula_1', 'sports_competition',
                              'soccer_2', 'swimming', 'poker_player', 'decoration_competition', 'climbing', 'club_1',
                              'riding_club', 'university_basketball']
        if self.dataset_type == 'spider' or self.dataset_type == 'cosql':
            self.etype = 'all'
        elif self.dataset_type == 'wikisql' or 'combine_multi':
            self.etype = 'match'

    def _post_process(self, dataset: Dataset, predictions: np.ndarray):
        gold_sql = dataset['sql']
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        gold_db = dataset['db_id']
        interaction_ids = dataset['interaction_id']

        if self.dataset_type == 'cosql':
            pre_ia_id, g_ia_db = None, None
            g_ia_one, p_ia_one = [], []
            gold_interactions, pred_interactions, gold_interaction_dbs = [], [], []
            for i, ia_id in enumerate(interaction_ids):
                if ia_id != pre_ia_id and pre_ia_id is not None:
                    gold_interactions.append(g_ia_one)
                    pred_interactions.append(p_ia_one)
                    gold_interaction_dbs.append(g_ia_db)
                    g_ia_one, p_ia_one = [gold_sql[i]], [predictions[i]]
                    g_ia_db = gold_db[i]
                else:
                    g_ia_one.append(gold_sql[i])
                    p_ia_one.append(predictions[i])
                    g_ia_db = gold_db[i]
                pre_ia_id = ia_id
            gold_interactions.append(g_ia_one)
            pred_interactions.append(p_ia_one)
            gold_interaction_dbs.append(g_ia_db)

            predictions = pred_interactions
            gold_sql = gold_interactions
            gold_db = gold_interaction_dbs

        return EvalPrediction(predictions=predictions, gold_sql=gold_sql, gold_db=gold_db)

    def _compute_metrics(self, eval_prediction: EvalPrediction, etype: str, in_prediction: bool = False):
        predictions, gold_sql, gold_db = eval_prediction
        if self.dataset_type == 'combine_multi':
            if not in_prediction:
                spider_predictions, spider_gold_sql, spider_gold_db = [], [], []
                wikisql_predictions, wikisql_gold_sql, wikisql_gold_db = [], [], []
                for p, g, db in zip(predictions, gold_sql, gold_db):
                    if db in self.spider_db_ids:
                        spider_predictions.append(p)
                        spider_gold_sql.append(g)
                        spider_gold_db.append(db)
                    else:
                        wikisql_predictions.append(p)
                        wikisql_gold_sql.append(g)
                        wikisql_gold_db.append(db)
                spider_match_score, _ = spider_evaluate(gold_sql=spider_gold_sql, gold_db=spider_gold_db,
                                                        predict=spider_predictions, etype='match')
                wikisql_match_score, _ = wikisql_evaluate(gold_sql=wikisql_gold_sql, gold_db=wikisql_gold_db,
                                                          predict=wikisql_predictions, etype='match')
                match_score = (len(spider_gold_db) * spider_match_score + len(
                    wikisql_gold_db) * wikisql_match_score) / (len(spider_gold_db) + len(wikisql_gold_db))
                metrics = {'eval_exact_match': match_score}
                eval_results = None

            else:
                if gold_db[0] in self.spider_db_ids:
                    evaluation_method = spider_evaluate
                else:
                    evaluation_method = wikisql_evaluate
                match_score, eval_results = evaluation_method(gold_sql=gold_sql, gold_db=gold_db,
                                                              predict=predictions, etype='match')
                metrics = {'eval_exact_match': match_score}

        else:
            if self.dataset_type == 'spider':
                evaluation_method = spider_evaluate
            elif self.dataset_type == 'wikisql':
                evaluation_method = wikisql_evaluate
            elif self.dataset_type == 'cosql':
                evaluation_method = cosql_evaluate
            else:
                raise NotImplementedError

            if etype == 'match':
                match_score, eval_results = evaluation_method(gold_sql=gold_sql, gold_db=gold_db,
                                                              predict=predictions, etype='match')
                metrics = {'eval_exact_match': match_score}
            else:
                exec_score, match_score, eval_results = evaluation_method(gold_sql=gold_sql, gold_db=gold_db,
                                                                          predict=predictions, etype='all')
                metrics = {'eval_exec': exec_score, 'eval_exact_match': match_score}

        return metrics, eval_results

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "test",
            **gen_kwargs
    ):

        gen_kwargs = gen_kwargs.copy()
        gen_kwargs["max_length"] = 256
        self._gen_kwargs = gen_kwargs

        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        self.compute_metrics = compute_metrics

        total_batch_size = self.args.eval_batch_size * self.args.world_size

        eval_preds = self._post_process(self.eval_dataset, output.predictions)
        metrics, _ = compute_metrics(eval_preds, etype=self.etype)

        with open(f'{self.output_dir}/eval_log.jsonl', 'a', encoding='utf-8') as writer:
            writer.write(json.dumps(metrics) + '\n')

        output.metrics.update(metrics)

        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(
            self,
            test_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            **gen_kwargs
    ):
        gen_kwargs = gen_kwargs.copy()
        gen_kwargs["max_length"] = 256
        self._gen_kwargs = gen_kwargs

        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(test_dataset)
        start_time = time.time()

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Prediction",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        self.compute_metrics = compute_metrics

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        preds = self._post_process(test_dataset, output.predictions)
        pred_metric, eval_results = compute_metrics(preds, in_prediction=True, etype=self.etype)
        output.metrics.update(pred_metric)

        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output, eval_results


class SubTrainer_qwen(Trainer):
    def __init__(self, dataset_type: str, **kwargs):
        super().__init__(**kwargs)
        self.compute_metrics = self._compute_metrics
        self.output_dir = self.args.output_dir
        self.dataset_type = dataset_type
        self.spider_db_ids = ['wine_1', 'game_injury', 'assets_maintenance', 'county_public_safety', 'farm', 'wedding',
                              'architecture', 'machine_repair', 'battle_death', 'debate', 'behavior_monitoring',
                              'perpetrator', 'college_2', 'csu_1', 'hospital_1', 'college_1', 'college_3', 'flight_2',
                              'flight_4', 'ship_mission', 'ship_1', 'flight_1', 'railway', 'aircraft', 'pilot_record',
                              'flight_company', 'train_station', 'restaurants', 'yelp', 'restaurant_1', 'theme_gallery',
                              'cre_Theme_park', 'roller_coaster', 'inn_1', 'apartment_rentals', 'coffee_shop',
                              'museum_visit', 'film_rank', 'imdb', 'program_share', 'entertainment_awards', 'cinema',
                              'tvshow', 'movie_1', 'insurance_policies', 'loan_1', 'small_bank_1', 'solvency_ii',
                              'insurance_fnol', 'insurance_and_eClaims', 'real_estate_properties',
                              'tracking_software_problems', 'network_2', 'allergy_1', 'protein_institute',
                              'dog_kennels', 'network_1', 'medicine_enzyme_interaction', 'device', 'station_weather',
                              'pets_1', 'twitter_1', 'storm_record', 'browser_web', 'wta_1', 'match_season', 'wrestler',
                              'gymnast', 'bike_1', 'body_builder', 'race_track', 'formula_1', 'sports_competition',
                              'soccer_2', 'swimming', 'poker_player', 'decoration_competition', 'climbing', 'club_1',
                              'riding_club', 'university_basketball']
        if self.dataset_type == 'spider' or self.dataset_type == 'cosql':
            self.etype = 'all'
        elif self.dataset_type == 'wikisql' or 'combine_multi':
            self.etype = 'match'

    def _post_process(self, dataset: Dataset, predictions: np.ndarray):
        gold_sql = dataset['sql']
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        gold_db = dataset['db_id']
        interaction_ids = dataset['interaction_id']

        if self.dataset_type == 'cosql':
            pre_ia_id, g_ia_db = None, None
            g_ia_one, p_ia_one = [], []
            gold_interactions, pred_interactions, gold_interaction_dbs = [], [], []
            for i, ia_id in enumerate(interaction_ids):
                if ia_id != pre_ia_id and pre_ia_id is not None:
                    gold_interactions.append(g_ia_one)
                    pred_interactions.append(p_ia_one)
                    gold_interaction_dbs.append(g_ia_db)
                    g_ia_one, p_ia_one = [gold_sql[i]], [predictions[i]]
                    g_ia_db = gold_db[i]
                else:
                    g_ia_one.append(gold_sql[i])
                    p_ia_one.append(predictions[i])
                    g_ia_db = gold_db[i]
                pre_ia_id = ia_id
            gold_interactions.append(g_ia_one)
            pred_interactions.append(p_ia_one)
            gold_interaction_dbs.append(g_ia_db)

            predictions = pred_interactions
            gold_sql = gold_interactions
            gold_db = gold_interaction_dbs

        return EvalPrediction(predictions=predictions, gold_sql=gold_sql, gold_db=gold_db)

    def _compute_metrics(self, eval_prediction: EvalPrediction, etype: str, in_prediction: bool = False):
        predictions, gold_sql, gold_db = eval_prediction
        if self.dataset_type == 'combine_multi':
            if not in_prediction:
                spider_predictions, spider_gold_sql, spider_gold_db = [], [], []
                wikisql_predictions, wikisql_gold_sql, wikisql_gold_db = [], [], []
                for p, g, db in zip(predictions, gold_sql, gold_db):
                    if db in self.spider_db_ids:
                        spider_predictions.append(p)
                        spider_gold_sql.append(g)
                        spider_gold_db.append(db)
                    else:
                        wikisql_predictions.append(p)
                        wikisql_gold_sql.append(g)
                        wikisql_gold_db.append(db)
                spider_match_score, _ = spider_evaluate(gold_sql=spider_gold_sql, gold_db=spider_gold_db,
                                                        predict=spider_predictions, etype='match')
                wikisql_match_score, _ = wikisql_evaluate(gold_sql=wikisql_gold_sql, gold_db=wikisql_gold_db,
                                                          predict=wikisql_predictions, etype='match')
                match_score = (len(spider_gold_db) * spider_match_score + len(
                    wikisql_gold_db) * wikisql_match_score) / (len(spider_gold_db) + len(wikisql_gold_db))
                metrics = {'eval_exact_match': match_score}
                eval_results = None

            else:
                if gold_db[0] in self.spider_db_ids:
                    evaluation_method = spider_evaluate
                else:
                    evaluation_method = wikisql_evaluate
                match_score, eval_results = evaluation_method(gold_sql=gold_sql, gold_db=gold_db,
                                                              predict=predictions, etype='match')
                metrics = {'eval_exact_match': match_score}

        else:
            if self.dataset_type == 'spider':
                evaluation_method = spider_evaluate
            elif self.dataset_type == 'wikisql':
                evaluation_method = wikisql_evaluate
            elif self.dataset_type == 'cosql':
                evaluation_method = cosql_evaluate
            else:
                raise NotImplementedError

            if etype == 'match':
                match_score, eval_results = evaluation_method(gold_sql=gold_sql, gold_db=gold_db,
                                                              predict=predictions, etype='match')
                metrics = {'eval_exact_match': match_score}
            else:
                exec_score, match_score, eval_results = evaluation_method(gold_sql=gold_sql, gold_db=gold_db,
                                                                          predict=predictions, etype='all')
                metrics = {'eval_exec': exec_score, 'eval_exact_match': match_score}

        return metrics, eval_results

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "test",
            **gen_kwargs
    ):

        gen_kwargs = gen_kwargs.copy()
        gen_kwargs["max_length"] = 256
        self._gen_kwargs = gen_kwargs

        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        self.compute_metrics = compute_metrics

        total_batch_size = self.args.eval_batch_size * self.args.world_size

        eval_preds = self._post_process(self.eval_dataset, output.predictions)
        metrics, _ = compute_metrics(eval_preds, etype=self.etype)

        with open(f'{self.output_dir}/eval_log.jsonl', 'a', encoding='utf-8') as writer:
            writer.write(json.dumps(metrics) + '\n')

        output.metrics.update(metrics)

        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(
            self,
            test_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            **gen_kwargs
    ):
        gen_kwargs = gen_kwargs.copy()
        gen_kwargs["max_length"] = 256
        self._gen_kwargs = gen_kwargs

        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(test_dataset)
        start_time = time.time()

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Prediction",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        self.compute_metrics = compute_metrics

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        preds = self._post_process(test_dataset, output.predictions)
        pred_metric, eval_results = compute_metrics(preds, in_prediction=True, etype=self.etype)
        output.metrics.update(pred_metric)

        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output, eval_results


class FinetunePredictorTrainer_kl(PredictorTrainer):
    def __init__(self, args):
        super().__init__(args)
        set_seed(self.args.seed)
        self.trainer = None
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_args, self.data_args, self.training_args = parser.parse_dict(vars(args))

        try:
            nltk.data.find("tokenizers/punkt")
        except (LookupError, OSError):
            if is_offline_mode():
                raise LookupError(
                    "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
                )
            with FileLock(".lock") as lock:
                nltk.download("punkt", quiet=True)

        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
        log_level = self.training_args.get_process_log_level()
        self.logger.setLevel(log_level)
        datasets.utils.logging.set_verbosity(log_level)
        transformers.logging.set_verbosity(log_level)
        transformers.logging.enable_default_handler()
        transformers.logging.enable_explicit_format()

        self.logger.warning(
            f"Process rank: {self.training_args.local_rank}, device: {self.training_args.device}, "
            f"n_gpu: {self.training_args.n_gpu}, distributed training: {bool(self.training_args.local_rank != -1)}, "
            f"16-bits training: {self.training_args.fp16}"
        )

        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
        f'train/plms/{self.args.backbone_plm}',
        cache_dir=self.model_args.cache_dir,
        use_fast=self.model_args.use_fast_tokenizer,
        revision=self.model_args.model_revision,
        use_auth_token=True if self.model_args.use_auth_token else None,
        )

        self.tokenizer.truncation_side = 'left'
        self.tokenizer.add_tokens(['<', '<='])

        self.generation_arguments = {
            'max_length': 256,
            'max_new_tokens': None,
            'min_length': 5,
            'temperature': 1.0,
            'do_sample': False,
            'top_k': 0,
            'top_p': 0.9,
            'repetition_penalty': 1.0,
            'num_beams': 4,
            'bad_words_ids': [[628], [198]]
        }


    def prepare_raw_datasets(self, dataset_dir: str, task_id: int, shuffle_tables: bool, shuffle_columns: bool):
        data_files = {
            'train': f'{dataset_dir}/task_{task_id}/train_seq2seq.jsonl',
            'val': f'{dataset_dir}/task_{task_id}/dev_seq2seq.jsonl',
            'test': f'{dataset_dir}/task_{task_id}/test_seq2seq.jsonl'
        }

        raw_datasets = {}
        for ds_name in data_files.keys():
            raw_datasets[ds_name] = {'text': [], 'sql': [], 'db_id': [], 'dataset_type': [], 'interaction_id': []}
            with open(data_files[ds_name], 'r', encoding='utf-8') as reader:
                for line in reader.readlines():
                    ex = json.loads(line)
                    # shuffle
                    if shuffle_tables:
                        text = ex['text']
                        parts = text.split("|")
                        assert len(parts) == 2, f"Error in splits:{text} "
                        db_data = parts[0].strip().split("; ")
                        
                        if db_data[-1].endswith(';'):
                            db_data[-1] = db_data[-1].rstrip(';')
                        question = parts[1].strip()
                        random.shuffle(db_data)
                        shuffle_tables = "; ".join(db_data) + ';'
                        new_text = shuffle_tables + " | " + question
                        # print(text)
                        # print()
                        # print(new_text)
                        raw_datasets[ds_name]['text'].append(new_text)
                    
                    else:
                        raw_datasets[ds_name]['text'].append(ex['text'])

                    raw_datasets[ds_name]['sql'].append(ex['sql'])
                    if 'example' in ex:
                        raw_datasets[ds_name]['db_id'].append(ex['example']['db_id'])
                    else:
                        raw_datasets[ds_name]['db_id'].append(ex['db_id'])
                    
                    if 'combine' in self.args.dataset_dir:
                        raw_datasets[ds_name]['dataset_type'].append(ex['dataset_name'])
                        raw_datasets[ds_name]['interaction_id'].append(None)
                    elif 'spider' in self.args.dataset_dir:
                        raw_datasets[ds_name]['dataset_type'].append('spider')
                        raw_datasets[ds_name]['interaction_id'].append(None)

                    elif 'cosql' in self.args.dataset_dir:
                        raw_datasets[ds_name]['dataset_type'].append('cosql')
                        if 'interaction_id' in ex['example'].keys():
                            raw_datasets[ds_name]['interaction_id'].append(ex['example']['interaction_id'])
                        else:
                            raw_datasets[ds_name]['interaction_id'].append(None)
                    else:
                        raise NotImplementedError

            # raw_datasets[ds_name] = datasets.Dataset.from_dict(raw_datasets[ds_name])
            raw_datasets[ds_name] = MyDataset_others(raw_datasets[ds_name], self.tokenizer)

        return raw_datasets
    
    def prepare_raw_datasets_kl(self, dataset_dir: str, task_id: int, shuffle_tables: bool, shuffle_columns: bool):
        data_files = {
            'train': f'{dataset_dir}/task_{task_id}/train_seq2seq.jsonl',
            'val': f'{dataset_dir}/task_{task_id}/dev_seq2seq.jsonl',
            'test': f'{dataset_dir}/task_{task_id}/test_seq2seq.jsonl'
        }

        raw_datasets = {}
        for ds_name in data_files.keys():
            if ds_name == 'train':
                raw_datasets[ds_name] = {'text': [], 'sql': [], 'db_id': [], 'dataset_type': [], 'interaction_id': [], 'domain_type': []}
            else:
                raw_datasets[ds_name] = {'text': [], 'sql': [], 'db_id': [], 'dataset_type': [], 'interaction_id': []}
            with open(data_files[ds_name], 'r', encoding='utf-8') as reader:
                for line in reader.readlines():
                    ex = json.loads(line)
                    # shuffle
                    if shuffle_tables:
                        text = ex['text']
                        parts = text.split("|")
                        assert len(parts) == 2, f"Error in splits:{text} "
                        db_data = parts[0].strip().split("; ")
                        if db_data[-1].endswith(';'):
                            db_data[-1] = db_data[-1].rstrip(';')
                        question = parts[1].strip()
                        random.shuffle(db_data)
                        shuffle_tables = "; ".join(db_data) + ';'
                        new_text = shuffle_tables + " | " + question
                        # print(text)
                        # print()
                        # print(new_text)
                        raw_datasets[ds_name]['text'].append(new_text)
                    
                    else:
                        raw_datasets[ds_name]['text'].append(ex['text'])

                    raw_datasets[ds_name]['sql'].append(ex['sql'])
                    if 'example' in ex:
                        raw_datasets[ds_name]['db_id'].append(ex['example']['db_id'])
                    else:
                        raw_datasets[ds_name]['db_id'].append(ex['db_id'])
                    
                    if 'combine' in self.args.dataset_dir:
                        raw_datasets[ds_name]['dataset_type'].append(ex['dataset_name'])
                        raw_datasets[ds_name]['interaction_id'].append(None)
                        if ds_name == 'train':
                            if ex['type'] == 'ORI':
                                raw_datasets[ds_name]['domain_type'].append(1)
                            elif ex['type'] == 'ID':
                                raw_datasets[ds_name]['domain_type'].append(2)
                            elif ex['type'] == 'OOD':
                                raw_datasets[ds_name]['domain_type'].append(3)
                            else:
                                assert 1 == 2 

                    elif 'spider' in self.args.dataset_dir:
                        raw_datasets[ds_name]['dataset_type'].append('spider')
                        raw_datasets[ds_name]['interaction_id'].append(None)
                        if ds_name == 'train':
                            if ex['type'] == 'ORI':
                                raw_datasets[ds_name]['domain_type'].append(1)
                            elif ex['type'] == 'ID':
                                raw_datasets[ds_name]['domain_type'].append(2)
                            elif ex['type'] == 'OOD':
                                raw_datasets[ds_name]['domain_type'].append(3)
                            else:
                                assert 1 == 2 
                        
                    elif 'cosql' in self.args.dataset_dir:
                        raw_datasets[ds_name]['dataset_type'].append('cosql')
                        if 'interaction_id' in ex['example'].keys():
                            raw_datasets[ds_name]['interaction_id'].append(ex['example']['interaction_id'])
                        else:
                            raw_datasets[ds_name]['interaction_id'].append(None)
                    else:
                        raise NotImplementedError
            
            # if ds_name == 'train':
            #     raw_datasets[ds_name] = MyDataset_train(raw_datasets[ds_name], self.tokenizer)
            # else:
            #     raw_datasets[ds_name] = MyDataset_others(raw_datasets[ds_name], self.tokenizer) 
            
            if self.args.fast_debug == 1:
                raw_datasets_debug = {}
                for keys, values in raw_datasets[ds_name].items():
                    raw_datasets_debug[keys] = values[:20]
                raw_datasets[ds_name] = datasets.Dataset.from_dict(raw_datasets_debug)
            else:
                raw_datasets[ds_name] = datasets.Dataset.from_dict(raw_datasets[ds_name])

        return raw_datasets
    
    def prepare_raw_datasets_full(self, dataset_dir: str, task_id: int):
        """
        """
        # 根据当前Task_id读取历史训练数据
        raw_datasets =  {
                            'train': {'text': [], 'sql': [], 'db_id': [], 'dataset_type': [], 'interaction_id': []},
                            'val': {'text': [], 'sql': [], 'db_id': [], 'dataset_type': [], 'interaction_id': []},
                            'test': {'text': [], 'sql': [], 'db_id': [], 'dataset_type': [], 'interaction_id': []}
                        }

        for i in range(task_id+1):
            log_data_files = {
                                'train': f'{dataset_dir}/task_{i}/train_seq2seq.jsonl',
                                'val': f'{dataset_dir}/task_{i}/dev_seq2seq.jsonl',
                                'test': f'{dataset_dir}/task_{i}/test_seq2seq.jsonl'
                              }
            
            with open(log_data_files['train'], "r", encoding="utf-8") as reader:
                for line in reader.readlines():
                    ex = json.loads(line)
                    raw_datasets['train']['text'].append(ex['text'])
                    raw_datasets['train']['sql'].append(ex['sql'])
                    raw_datasets['train']['dataset_type'].append(None)
                    raw_datasets['train']['interaction_id'].append(None)
                    if 'example' in ex:
                        raw_datasets['train']['db_id'].append(ex['example']['db_id'])
                    else:
                        raw_datasets['train']['db_id'].append(ex['db_id'])
            
            if i == task_id:
                for split in ['val', 'test']:
                    with open(log_data_files[split], 'r', encoding='utf-8') as reader1:
                        for line1 in reader1.readlines():
                            ex1 = json.loads(line1)
                            raw_datasets[split]['text'].append(ex1['text'])
                            raw_datasets[split]['sql'].append(ex1['sql'])
                            raw_datasets[split]['dataset_type'].append(None)
                            raw_datasets[split]['interaction_id'].append(None)
                            if 'example' in ex:
                                raw_datasets[split]['db_id'].append(ex1['example']['db_id'])
                            else:
                                raw_datasets[split]['db_id'].append(ex1['db_id'])
 
        print(f"Task_{task_id}_train_{len(raw_datasets['train']['text'])}条")
        print(f"Task_{task_id}_val_{len(raw_datasets['val']['text'])}条")
        print(f"Task_{task_id}_test_{len(raw_datasets['test']['text'])}条")

        for split in raw_datasets.keys():
            raw_datasets[split] = datasets.Dataset.from_dict(raw_datasets[split])
        
        return raw_datasets

    def train_step(self, 
                   batch, 
                   task_id,
                   model, 
                   past_model=None):

        input_ids = batch["input_ids"].to(self.device)
        input_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        if self.args.aug == True:
            types = batch["domain_type"]

        labels[labels[:, :] == self.tokenizer.pad_token_id] = -100
        
        outputs = model(input_ids=input_ids, attention_mask=input_mask, labels=labels, return_dict=True)
        loss_ce = outputs.loss
        
        if task_id > 0 and self.args.aug == True:
            loss_kl = torch.tensor(0.0).to(self.device) 
            with torch.no_grad():
                teacher_outputs = past_model(input_ids=input_ids, attention_mask=input_mask, labels=labels, return_dict=True)
                for i, type in enumerate(types):
                    if type == "OOD":
                        #计算KL散度
                        outputs_logits = outputs.logits
                        teacher_outputs_logits = teacher_outputs.logits
                        kl_loss = F.kl_div(F.log_softmax(outputs_logits[i], dim=-1), F.softmax(teacher_outputs_logits[i], dim=-1), reduction='batchmean')
                        loss_kl += kl_loss

            total_loss = (1 - self.args.alpha) * loss_ce + self.args.alpha * loss_kl
        else:
            total_loss = loss_ce

        return total_loss

    def get_meta(self, dataset):
        """
        get lists of db_ids, gold SQL queries and interaction_ids (CoSQL) for examples in the dataloader
        :param dataloader: dataloader for evaluation/prediction
        :return: a list of db_ids, a list of gold SQL queries
        """
        db_ids, sqls, interaction_ids = [], [], []
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        for batch in dataloader:
                db_ids.append(batch["db_id"])
                sqls.append(batch["sql"])
        
        db_ids_ = [item[0] for item in db_ids]
        sqls_ = [item[0] for item in sqls]

        return db_ids_, sqls_, interaction_ids

    def evaluate(self, model, dataloader, val_dataset):
        """
        evaluate generated SQL queries' match/execution accuracies
        :param prompt_model: loaded PLM & soft prompt template
        :param dataloader: dataloader for evaluation
        :return: lists of execution scores, match scores, evaluation results, generated SQL queries
        """
        generated_sqls = []
        gt_dbs, gt_sqls, interaction_ids = self.get_meta(val_dataset)
        exec_score, match_score = None, None
        with torch.no_grad():
            model.eval()
            for step, batch in enumerate(tqdm(dataloader, desc='evaluation')):
                # if self.args.use_cuda:
                #     inputs = inputs.cuda()
                input_ids = batch["input_ids"].to(self.device)
                input_mask = batch["attention_mask"].to(self.device)
                # target_ids = batch["target_ids"].to(self.device)
                # target_mask = batch["target_mask"].to(self.device)

                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    **self.generation_arguments
                )                
                for i in range(outputs.shape[0]):
                    generated_sql = self.tokenizer.decode(outputs[i], skip_special_tokens=True) 
                    generated_sqls.append(generated_sql)

            if self.args.dataset.startswith('spider'):
                exec_score, match_score, eval_results = spider_evaluate(gt_sqls, gt_dbs, generated_sqls, 'all')

            elif self.args.dataset.startswith('combine'):
                dataset_name = dataloader.raw_dataset[0].meta['dataset_name']
                if dataset_name == 'spider':
                    match_score, eval_results = spider_evaluate(gt_sqls, gt_dbs, generated_sqls, 'match')
                elif dataset_name == 'wikisql':
                    match_score, eval_results = wikisql_evaluate(gt_sqls, gt_dbs, generated_sqls, 'match')

            elif self.args.dataset.startswith('cosql'):
                pre_ia_id, g_ia_db = None, None
                g_ia_one, p_ia_one = [], []
                gold_interactions, pred_interactions, gold_interaction_dbs = [], [], []
                for i, ia_id in enumerate(interaction_ids):
                    if ia_id != pre_ia_id and pre_ia_id is not None:
                        gold_interactions.append(g_ia_one)
                        pred_interactions.append(p_ia_one)
                        gold_interaction_dbs.append(g_ia_db)
                        g_ia_one, p_ia_one = [gt_sqls[i]], [generated_sqls[i]]
                        g_ia_db = gt_dbs[i]
                    else:
                        g_ia_one.append(gt_sqls[i])
                        p_ia_one.append(generated_sqls[i])
                        g_ia_db = gt_dbs[i]
                    pre_ia_id = ia_id
                gold_interactions.append(g_ia_one)
                pred_interactions.append(p_ia_one)
                gold_interaction_dbs.append(g_ia_db)

                gt_sqls = gold_interactions
                gt_dbs = gold_interaction_dbs

                exec_score, match_score, eval_results = cosql_evaluate(gt_sqls, gt_dbs, pred_interactions, 'all')
            else:
                raise NotImplementedError

        return exec_score, match_score, eval_results, generated_sqls

    def train(self, task_id):
        torch.cuda.empty_cache()
        if task_id == 0:
            model_name_or_path = f'train/plms/{self.args.backbone_plm}'
        else:
            model_name_or_path = f'{self.args.output_dir}/task_{task_id-1}'

        self.training_args.output_dir = f'{self.args.output_dir}/task_{task_id}'

        last_checkpoint = None
        if os.path.isdir(self.training_args.output_dir) and \
                self.training_args.do_train and not self.training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(self.training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(self.training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({self.training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and self.training_args.resume_from_checkpoint is None:
                self.logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        if self.args.full_data:
            raw_datasets = self.prepare_raw_datasets_full(self.args.dataset_dir, task_id)
        elif self.args.aug:
            raw_datasets = self.prepare_raw_datasets_kl(self.args.dataset_dir, task_id, shuffle_tables=self.args.shuffle_tables, shuffle_columns=self.args.shuffle_columns)
        else:
            raw_datasets = self.prepare_raw_datasets(self.args.dataset_dir, task_id, shuffle_tables=self.args.shuffle_tables, shuffle_columns=self.args.shuffle_columns)

        config = AutoConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else model_name_or_path,
            # f'train/plms/{self.args.backbone_plm}',
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name if self.model_args.tokenizer_name else model_name_or_path,
            # f'train/plms/{self.args.backbone_plm}',
            cache_dir=self.model_args.cache_dir,
            use_fast=self.model_args.use_fast_tokenizer,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )

        tokenizer.truncation_side = 'left'
        tokenizer.add_tokens(['<', '<='])

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            from_tf=False,
            config=config,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )

        model.resize_token_embeddings(len(tokenizer))

        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        if (
                hasattr(model.config, "max_position_embeddings")
                and model.config.max_position_embeddings < self.data_args.max_source_length
        ):
            if self.model_args.resize_position_embeddings is None:
                self.logger.warning(
                    f"Increasing the model's number of position embedding vectors from "
                    f"{model.config.max_position_embeddings} to {self.data_args.max_source_length}."
                )
                model.resize_position_embeddings(self.data_args.max_source_length)
            elif self.model_args.resize_position_embeddings:
                model.resize_position_embeddings(self.data_args.max_source_length)
            else:
                raise ValueError(
                    f"`--max_source_length` is set to {self.data_args.max_source_length}, but the model only has "
                    f"{model.config.max_position_embeddings} position encodings. Consider either reducing "
                    f"`--max_source_length` to {model.config.max_position_embeddings} or to automatically resize "
                    f"the model's position encodings by passing `--resize_position_embeddings`."
                )

        # 加载Teacher
        # teacher_model = copy.deepcopy(model)
        if task_id > 0:
            teacher_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                from_tf=False,
                config=config,
                cache_dir=self.model_args.cache_dir,
                revision=self.model_args.model_revision,
                use_auth_token=True if self.model_args.use_auth_token else None,
            )
            teacher_model.resize_token_embeddings(len(tokenizer))
            # teacher_model.to(self.device)
            teacher_model.eval()
        
        else:
            teacher_model = None


        prefix = self.data_args.source_prefix if self.data_args.source_prefix is not None else ""

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = raw_datasets['test'].column_names
        text_column, summary_column = column_names[0], column_names[1]

        # domain_type
        type_column = raw_datasets['train'].column_names[-1]

        # Temporarily set max_target_length for training.
        max_target_length = self.data_args.max_target_length
        padding = self.data_args.pad_to_max_length

        if self.training_args.label_smoothing_factor > 0 and \
                not hasattr(model, "prepare_decoder_input_ids_from_labels"):
            self.logger.warning(
                f"label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined "
                f"for `{model.__class__.__name__}`. "
                f"This will lead to loss being calculated twice and will take up more memory"
            )

        def preprocess_function(examples):
            # remove pairs where at least one record is None
            inputs, targets = [], []
            for i in range(len(examples[text_column])):
                if examples[text_column][i] is not None and examples[summary_column][i] is not None:
                    inputs.append(examples[text_column][i])
                    targets.append(examples[summary_column][i])

            inputs = [prefix + inp for inp in inputs]
            model_inputs = tokenizer(inputs, max_length=self.data_args.max_source_length, padding=padding,
                                     truncation=True)
            # Set up the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)
            model_inputs["labels"] = labels["input_ids"]

            return model_inputs
        
        def preprocess_function_kl(examples):
            # remove pairs where at least one record is None
            inputs, targets, types = [], [], []
            for i in range(len(examples[text_column])):
                if examples[text_column][i] is not None and examples[summary_column][i] is not None:
                    inputs.append(examples[text_column][i])
                    targets.append(examples[summary_column][i])
                    types.append(examples[type_column][i])

            inputs = [prefix + inp for inp in inputs]
            model_inputs = tokenizer(inputs, max_length=self.data_args.max_source_length, padding=padding,
                                     truncation=True)
            # Set up the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)
            model_inputs["labels"] = labels["input_ids"]

            # add domain_type
            # model_inputs['type'] = types 

            return model_inputs

        if self.training_args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset")
            train_dataset = raw_datasets["train"]

            len_epoch_iterator = math.ceil(len(train_dataset) / (self.args.per_device_train_batch_size *
                                                                 self.n_gpu))
            steps_per_epoch = math.ceil(len_epoch_iterator / self.args.gradient_accumulation_steps)

            print(f'\nsteps_per_epoch: {steps_per_epoch}\n')

            self.training_args.eval_steps = steps_per_epoch * 5
            self.training_args.save_steps = self.training_args.eval_steps
            self.training_args.eval_delay = steps_per_epoch * 20

            with self.training_args.main_process_first(desc="train dataset map pre-processing"):
                train_dataset = train_dataset.map(
                    preprocess_function_kl,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset"
                )

        if self.training_args.do_eval:
            max_target_length = self.data_args.val_max_target_length
            if "val" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["val"]
            if self.data_args.max_eval_samples is not None:
                eval_dataset = eval_dataset.select(range(self.data_args.max_eval_samples))
            with self.training_args.main_process_first(desc="validation dataset map pre-processing"):
                eval_dataset = eval_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                )

        if self.training_args.do_predict:
            max_target_length = self.data_args.val_max_target_length
            if "test" not in raw_datasets:
                raise ValueError("--do_predict requires a test dataset")
            predict_dataset = raw_datasets["test"]
            with self.training_args.main_process_first(desc="prediction dataset map pre-processing"):
                predict_dataset = predict_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on prediction dataset",
                )

            # Data collator
        label_pad_token_id = -100 if self.data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if self.training_args.fp16 else None,
        )

        # Initialize our Trainer
        trainer_args = {
            'model': model,
            'args': self.training_args,
            'train_dataset': train_dataset if self.training_args.do_train else None,
            'eval_dataset': eval_dataset if self.training_args.do_eval else None,
            'tokenizer': tokenizer,
            'data_collator': data_collator,
            'compute_metrics': None,
            'callbacks': [EarlyStoppingCallback(early_stopping_patience=5)],
        }

        if self.training_args.dataset.startswith('combine'):
            if not 'multi' in self.args.dataset:
                if raw_datasets['test']['dataset_type'][0] == 'wikisql':
                    dataset_type = 'wikisql'
                    trainer = SubTrainer(dataset_type=dataset_type, **trainer_args)
                elif raw_datasets['test']['dataset_type'][0] == 'spider':
                    dataset_type = 'spider'
                    trainer = SubTrainer(dataset_type=dataset_type, **trainer_args)
                else:
                    raise NotImplementedError
            else:
                dataset_type = 'combine_multi'
                trainer = SubTrainer(dataset_type=dataset_type, **trainer_args)

        elif self.training_args.dataset.startswith('spider'):
            dataset_type = 'spider'
            if task_id == 0:
                trainer = SubTrainer(dataset_type=dataset_type, **trainer_args)
            else:
                # 增加KL的Subtrainer
                trainer = SubTrainer_kl(
                                        dataset_type=dataset_type, 
                                        task_id=task_id, 
                                        temperature=self.args.temperature, 
                                        alpha=self.args.alpha, 
                                        teacher_model=teacher_model, 
                                        **trainer_args
                                        )


        elif self.training_args.dataset.startswith('cosql'):
            dataset_type = 'cosql'
            trainer = SubTrainer(dataset_type, **trainer_args)
        else:
            raise NotImplementedError

        print(f'\nDataset Type: {dataset_type}', flush=True)
        print(f'Backbone Pretrained Model: {self.args.backbone_plm}\n', flush=True)

        if self.training_args.do_train:
            checkpoint = None
            if self.training_args.resume_from_checkpoint is not None:
                checkpoint = self.training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model()  # Saves the tokenizer too for easy upload

            metrics = train_result.metrics
            max_train_samples = (
                self.data_args.max_train_samples if self.data_args.max_train_samples is not None else len(
                    train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        # Evaluation
        max_length = (
            self.training_args.generation_max_length
            if self.training_args.generation_max_length is not None
            else self.data_args.val_max_target_length
        )
        num_beams = self.data_args.num_beams if self.data_args.num_beams is not None else self.training_args.generation_num_beams

        if self.training_args.do_predict:
            self.logger.info("*** Predict ***")
            output, eval_results = trainer.predict(
                predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
            )

            metrics = output.metrics
            max_predict_samples = (
                self.data_args.max_predict_samples if self.data_args.max_predict_samples is not None else len(
                    predict_dataset)
            )
            metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
            predictions = tokenizer.batch_decode(
                output.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]

            # save metrics & predict results
            output_spider_result_file = os.path.join(self.training_args.output_dir, "eval_results.txt")
            with open(output_spider_result_file, "w", encoding='utf-8') as writer:
                writer.write("\n".join(eval_results))
            output_prediction_file = os.path.join(self.training_args.output_dir, "generated_predictions.txt")
            with open(output_prediction_file, "w", encoding='utf-8') as writer:
                writer.write("\n".join(predictions))

        # delete checkpoints
        for name in os.listdir(self.training_args.output_dir):
            if os.path.isdir(os.path.join(self.training_args.output_dir, name)):
                shutil.rmtree(os.path.join(self.training_args.output_dir, name))


    def evaluate_continual_learning_metrics(self, do_wo_stu_ablation: bool = False):
        def get_task_size(data_dir: str, task_num: int):
            task_size_list = []
            for i in range(task_num):
                with open(f'{data_dir}/task_{i}/test_seq2seq.jsonl', 'r') as reader:
                    size = len(reader.readlines())
                task_size_list.append(size)
            return task_size_list

        def compute_acc(output_dir: str, task_id: int, data_dir: str, task_num: int):
            task_size_list = get_task_size(data_dir, task_num)
            acc_a, acc_w = 0, 0
            acc_a_joint, acc_w_joint = 0, 0  # used for CoSQL joint_all accuracy
            for t_id in range(task_id + 1):
                with open(f'{output_dir}/metrics/task_{task_id}/task_{t_id}/eval_results.txt', 'r') as reader:
                    for line in reader.readlines():
                        if line.startswith('exact match'):
                            if self.args.dataset.startswith('spider') or self.args.dataset.startswith('combine'):
                                acc = float(line.strip()[-5:])
                                break
                            elif self.args.dataset.startswith('cosql'):
                                acc = float(line.strip()[-26: -21])
                                acc_joint = float(line.strip()[-5:])
                                acc_a_joint += acc_joint
                                acc_w_joint += acc_joint * task_size_list[t_id]
                            break

                    acc_a += acc
                    acc_w += acc * task_size_list[t_id]

            acc_a /= task_id + 1
            acc_w /= sum(task_size_list[:task_id + 1])
            acc_a_joint /= task_id + 1
            acc_w_joint /= sum(task_size_list[:task_id + 1])

            if self.args.dataset.startswith('spider') or self.args.dataset.startswith('combine'):
                return acc_a, acc_w
            elif self.args.dataset.startswith('cosql'):
                return acc_a, acc_w, acc_a_joint, acc_w_joint

        def compute_bwt(output_dir, task_id):
            bwt = 0
            bwt_joint = 0
            for t_id in range(task_id):
                with open(f'{output_dir}/metrics/task_{task_id}/task_{t_id}/eval_results.txt', 'r') as reader:
                    for line in reader.readlines():
                        if line.startswith('exact match'):
                            if self.args.dataset.startswith('spider') or self.args.dataset.startswith('combine'):
                                backward_acc = float(line.strip()[-5:])
                            elif self.args.dataset.startswith('cosql'):
                                backward_acc = float(line.strip()[-26: -21])
                                backward_acc_joint = float(line.strip()[-5:])
                            break
                with open(f'{output_dir}/metrics/task_{t_id}/task_{t_id}/eval_results.txt', 'r') as reader:
                    for line in reader.readlines():
                        if line.startswith('exact match'):
                            if self.args.dataset.startswith('spider') or self.args.dataset.startswith('combine'):
                                original_acc = float(line.strip()[-5:])
                            elif self.args.dataset.startswith('cosql'):
                                original_acc = float(line.strip()[-26: -21])
                                original_acc_joint = float(line.strip()[-5:])
                            break
                bwt += (backward_acc - original_acc)
                if self.args.dataset.startswith('cosql'):
                    bwt_joint += (backward_acc_joint - original_acc_joint)

            bwt /= task_id

            if self.args.dataset.startswith('spider') or self.args.dataset.startswith('combine'):
                return bwt
            elif self.args.dataset.startswith('cosql'):
                bwt_joint /= task_id
                return bwt, bwt_joint

        def compute_fwt(output_dir, task_id):
            fwt = 0
            fwt_joint = 0
            for t_id in range(task_id + 1):
                with open(f'{output_dir}/metrics/task_{t_id}/task_{str(t_id + 1)}/eval_results.txt', 'r') as reader:
                    random_acc = 0.0
                    for line in reader.readlines():
                        if line.startswith('exact match'):
                            if self.args.dataset.startswith('spider') or self.args.dataset.startswith('combine'):
                                forward_acc = float(line.strip()[-5:])
                            elif self.args.dataset.startswith('cosql'):
                                forward_acc = float(line.strip()[-26:-21])
                                forward_acc_joint = float(line.strip()[-5:])
                            break
                fwt += (forward_acc - random_acc)
                if self.args.dataset.startswith('cosql'):
                    fwt_joint += (forward_acc_joint - random_acc)

            fwt /= (task_id + 1)

            if self.args.dataset.startswith('spider') or self.args.dataset.startswith('combine'):
                return fwt
            elif self.args.dataset.startswith('cosql'):
                fwt_joint /= (task_id + 1)
                return fwt, fwt_joint

        def prepare_raw_datasets_for_cl_evaluation(dataset_dir: str, cur_task_id: int, to_eval_task_id: int):

            if do_wo_stu_ablation:
                data_files = {
                    'test': f'{dataset_dir}/task_{cur_task_id}/backward/task_{to_eval_task_id}/test_seq2seq.jsonl'
                }
            else:
                data_files = {
                    'test': f'{dataset_dir}/task_{to_eval_task_id}/test_seq2seq.jsonl'
                }

            print(f'Load test file from ' + data_files['test'])

            raw_datasets = {}
            for ds_name in data_files.keys():
                raw_datasets[ds_name] = {
                    'text': [],
                    'sql': [],
                    'db_id': [],
                    'dataset_type': [],
                    'interaction_id': []
                }
                with open(data_files[ds_name], 'r', encoding='utf-8') as reader:
                    for line in reader.readlines():
                        ex = json.loads(line)
                        raw_datasets[ds_name]['text'].append(ex['text'])
                        raw_datasets[ds_name]['sql'].append(ex['sql'])
                        
                        if 'example' in ex:
                            raw_datasets[ds_name]['db_id'].append(ex['example']['db_id'])
                        else:
                            raw_datasets[ds_name]['db_id'].append(ex['db_id'])
                        # raw_datasets[ds_name]['db_id'].append(ex['example']['db_id'])
                        
                        if 'combine' in self.args.dataset_dir:
                            raw_datasets[ds_name]['dataset_type'].append(ex['example']['dataset_name'])
                            raw_datasets[ds_name]['interaction_id'].append(None)
                        elif 'spider' in self.args.dataset_dir:
                            raw_datasets[ds_name]['dataset_type'].append('spider')
                            raw_datasets[ds_name]['interaction_id'].append(None)
                        elif 'cosql' in self.args.dataset_dir:
                            raw_datasets[ds_name]['dataset_type'].append('cosql')
                            raw_datasets[ds_name]['interaction_id'].append(ex['example']['interaction_id'])
                        else:
                            raise NotImplementedError
                raw_datasets[ds_name] = datasets.Dataset.from_dict(raw_datasets[ds_name])

            return raw_datasets

        def predict(cur_task_id: int, to_eval_task_id: int):
            model_name_or_path = f'{self.args.output_dir}/task_{cur_task_id}'

            raw_datasets = prepare_raw_datasets_for_cl_evaluation(self.args.dataset_dir, cur_task_id, to_eval_task_id)

            config = AutoConfig.from_pretrained(
                model_name_or_path,
                cache_dir=self.model_args.cache_dir,
                revision=self.model_args.model_revision,
                use_auth_token=True if self.model_args.use_auth_token else None,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                cache_dir=self.model_args.cache_dir,
                use_fast=self.model_args.use_fast_tokenizer,
                revision=self.model_args.model_revision,
                use_auth_token=True if self.model_args.use_auth_token else None,
            )

            tokenizer.truncation_side = 'left'
            if self.args.dataset.startswith('cosql'):
                tokenizer.truncation_side = 'right'
            tokenizer.add_tokens(['<', '<='])

            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                from_tf=False,
                config=config,
                cache_dir=self.model_args.cache_dir,
                revision=self.model_args.model_revision,
                use_auth_token=True if self.model_args.use_auth_token else None,
            )

            model.resize_token_embeddings(len(tokenizer))

            if model.config.decoder_start_token_id is None:
                raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

            if (
                    hasattr(model.config, "max_position_embeddings")
                    and model.config.max_position_embeddings < self.data_args.max_source_length
            ):
                if self.model_args.resize_position_embeddings is None:
                    self.logger.warning(
                        f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                        f"to {self.data_args.max_source_length}."
                    )
                    model.resize_position_embeddings(self.data_args.max_source_length)
                elif self.model_args.resize_position_embeddings:
                    model.resize_position_embeddings(self.data_args.max_source_length)
                else:
                    raise ValueError(
                        f"`--max_source_length` is set to {self.data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                        f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                        "resize the model's position encodings by passing `--resize_position_embeddings`."
                    )

            prefix = self.data_args.source_prefix if self.data_args.source_prefix is not None else ""

            # Preprocessing the datasets.
            # We need to tokenize inputs and targets.
            column_names = raw_datasets['test'].column_names
            text_column, summary_column = column_names[0], column_names[1]

            # Temporarily set max_target_length for training.
            max_target_length = self.data_args.max_target_length
            padding = self.data_args.pad_to_max_length

            if self.training_args.label_smoothing_factor > 0 and not hasattr(model,
                                                                             "prepare_decoder_input_ids_from_labels"):
                self.logger.warning(
                    "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                    f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
                )

            def preprocess_function(examples):
                # remove pairs where at least one record is None

                inputs, targets = [], []
                for i in range(len(examples[text_column])):
                    if examples[text_column][i] is not None and examples[summary_column][i] is not None:
                        inputs.append(examples[text_column][i])
                        targets.append(examples[summary_column][i])

                inputs = [prefix + inp for inp in inputs]
                model_inputs = tokenizer(inputs, max_length=self.data_args.max_source_length, padding=padding,
                                         truncation=True)

                # Setup the tokenizer for targets
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

                model_inputs["labels"] = labels["input_ids"]
                return model_inputs

            max_target_length = self.data_args.val_max_target_length
            if "test" not in raw_datasets:
                raise ValueError("--do_predict requires a test dataset")
            predict_dataset = raw_datasets["test"]
            if self.data_args.max_predict_samples is not None:
                predict_dataset = predict_dataset.select(range(self.data_args.max_predict_samples))
            with self.training_args.main_process_first(desc="prediction dataset map pre-processing"):
                predict_dataset = predict_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=self.data_args.preprocessing_num_workers,
                    # remove_columns=column_names,
                    load_from_cache_file=not self.data_args.overwrite_cache,
                    desc="Running tokenizer on prediction dataset",
                )

            # Data collator
            label_pad_token_id = -100 if self.data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
            data_collator = DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=8 if self.training_args.fp16 else None,
            )

            trainer_args = {
                'model': model,
                'args': self.training_args,
                'tokenizer': tokenizer,
                'data_collator': data_collator,
                'compute_metrics': None,
                'callbacks': [EarlyStoppingCallback(early_stopping_patience=5)],
            }

            if self.training_args.dataset.startswith('combine'):
                if raw_datasets['test']['dataset_type'][0] == 'wikisql':
                    dataset_type = 'wikisql'
                    trainer = SubTrainer(dataset_type, **trainer_args)
                elif raw_datasets['test']['dataset_type'][0] == 'spider':
                    dataset_type = 'spider'
                    trainer = SubTrainer(dataset_type, **trainer_args)
                else:
                    raise NotImplementedError
            elif self.training_args.dataset.startswith('spider'):
                dataset_type = 'spider'
                trainer = SubTrainer(dataset_type, **trainer_args)
            elif raw_datasets['test']['dataset_type'][0] == 'cosql':
                dataset_type = 'cosql'
                trainer = SubTrainer(dataset_type, **trainer_args)
            else:
                raise NotImplementedError

            max_length = (
                self.training_args.generation_max_length
                if self.training_args.generation_max_length is not None
                else self.data_args.val_max_target_length
            )
            num_beams = self.data_args.num_beams if self.data_args.num_beams is not None else self.training_args.generation_num_beams

            self.logger.info("*** Predict ***")
            output, spider_results = trainer.predict(
                predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
            )

            metrics = output.metrics
            max_predict_samples = (
                self.data_args.max_predict_samples if self.data_args.max_predict_samples is not None else len(
                    predict_dataset)
            )
            metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
            predictions = tokenizer.batch_decode(
                output.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]

            # save metrics & predict results
            output_dir = f'{self.args.output_dir}/metrics/task_{cur_task_id}/task_{to_eval_task_id}'
            output_eval_result_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_result_file, 'w', encoding='utf-8') as writer:
                writer.write("\n".join(spider_results))
            output_prediction_file = os.path.join(output_dir, "generated_predictions.txt")
            with open(output_prediction_file, "w", encoding='utf-8') as writer:
                writer.write("\n".join(predictions))

        output_dir = f'{self.args.output_dir}/metrics'
        os.makedirs(output_dir, exist_ok=True)

        for cur_task_id in range(self.task_num):
            if not do_wo_stu_ablation:
                eval_range = range(cur_task_id + 1) if cur_task_id == self.task_num - 1 else range(cur_task_id + 2)
            else:
                eval_range = range(cur_task_id + 1)
            for to_eval_task_id in eval_range:
                os.makedirs(f'{output_dir}/task_{cur_task_id}/task_{to_eval_task_id}', exist_ok=True)
                if to_eval_task_id == cur_task_id:
                    source_dir = f'{self.args.output_dir}/task_{cur_task_id}/'
                    target_dir = f'{output_dir}/task_{cur_task_id}/task_{to_eval_task_id}'
                    os.makedirs(target_dir, exist_ok=True)
                    shutil.copy(f'{source_dir}/eval_results.txt', f'{target_dir}/eval_results.txt')
                    shutil.copy(f'{source_dir}/generated_predictions.txt', f'{target_dir}/generated_predictions.txt')
                else:
                    predict(cur_task_id, to_eval_task_id)

        results = {}

        data_dir = self.args.dataset_dir

        for task_id in range(self.task_num):
            results[f'task_{task_id}'] = {}
            if self.args.dataset.startswith('spider') or self.args.dataset.startswith('combine'):
                results[f'task_{task_id}']['acc_a'], results[f'task_{task_id}']['acc_w'] = \
                    compute_acc(self.args.output_dir, task_id, data_dir, self.task_num)
            elif self.args.dataset.startswith('cosql'):
                results[f'task_{task_id}']['acc_a'], results[f'task_{task_id}']['acc_w'], \
                    results[f'task_{task_id}']['acc_a_joint'], results[f'task_{task_id}']['acc_w_joint'] = \
                    compute_acc(self.args.output_dir, task_id, data_dir, self.task_num)

        for task_id in range(1, self.task_num):
            if self.args.dataset.startswith('spider') or self.args.dataset.startswith('combine'):
                results[f'task_{task_id}']['bwt'] = compute_bwt(self.args.output_dir, task_id)
            elif self.args.dataset.startswith('cosql'):
                results[f'task_{task_id}']['bwt'], results[f'task_{task_id}']['bwt_joint'] = \
                    compute_bwt(self.args.output_dir, task_id)

        if not do_wo_stu_ablation:
            for task_id in range(self.task_num - 1):
                if self.args.dataset.startswith('spider') or self.args.dataset.startswith('combine'):
                    results[f'task_{task_id}']['fwt'] = compute_fwt(self.args.output_dir, task_id)
                elif self.args.dataset.startswith('cosql'):
                    results[f'task_{task_id}']['fwt'], results[f'task_{task_id}']['fwt_joint'] = \
                        compute_fwt(self.args.output_dir, task_id)

        json.dump(results, open(f'{self.args.output_dir}/metrics.json', 'w', encoding='utf-8'), indent=4)

    def predict(self, cur_task_id: int, to_eval_task_id: int, stream=False):

        def prepare_raw_datasets_for_cl_evaluation(dataset_dir: str, cur_task_id: int, to_eval_task_id: int):


            data_files = {
                'test': f'{dataset_dir}/task_{to_eval_task_id}/test_seq2seq.jsonl'
            }

            print(f'Load test file from ' + data_files['test'])

            raw_datasets = {}
            for ds_name in data_files.keys():
                raw_datasets[ds_name] = {
                    'text': [],
                    'sql': [],
                    'db_id': [],
                    'dataset_type': [],
                    'interaction_id': []
                }
                with open(data_files[ds_name], 'r', encoding='utf-8') as reader:
                    for line in reader.readlines():
                        ex = json.loads(line)
                        raw_datasets[ds_name]['text'].append(ex['text'])
                        raw_datasets[ds_name]['sql'].append(ex['sql'])
                        if 'example' in ex:
                            raw_datasets[ds_name]['db_id'].append(ex['example']['db_id'])
                        else:
                            raw_datasets[ds_name]['db_id'].append(ex['db_id'])
                        # raw_datasets[ds_name]['db_id'].append(ex['example']['db_id'])
                        if 'combine' in self.args.dataset_dir:
                            raw_datasets[ds_name]['dataset_type'].append(ex['dataset_name'])
                            raw_datasets[ds_name]['interaction_id'].append(None)
                        elif 'spider' in self.args.dataset_dir:
                            raw_datasets[ds_name]['dataset_type'].append('spider')
                            raw_datasets[ds_name]['interaction_id'].append(None)
                        elif 'cosql' in self.args.dataset_dir:
                            raw_datasets[ds_name]['dataset_type'].append('cosql')
                            raw_datasets[ds_name]['interaction_id'].append(ex['example']['interaction_id'])
                        else:
                            raise NotImplementedError
                raw_datasets[ds_name] = datasets.Dataset.from_dict(raw_datasets[ds_name])

            return raw_datasets

        if os.path.exists(f'{self.args.output_dir}/task_{cur_task_id}/model'):
            model_name_or_path = f'{self.args.output_dir}/task_{cur_task_id}/model'
        else:
            model_name_or_path = f'{self.args.output_dir}/task_{cur_task_id}'

        raw_datasets = prepare_raw_datasets_for_cl_evaluation(self.args.dataset_dir, cur_task_id, to_eval_task_id)

        config = AutoConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name if self.model_args.tokenizer_name else model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            use_fast=self.model_args.use_fast_tokenizer,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )

        tokenizer.truncation_side = 'left'
        if self.args.dataset.startswith('cosql'):
            tokenizer.truncation_side = 'right'
        tokenizer.add_tokens(['<', '<='])

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path,
            from_tf=False,
            config=config,
            cache_dir=self.model_args.cache_dir,
            revision=self.model_args.model_revision,
            use_auth_token=True if self.model_args.use_auth_token else None,
        )

        model.resize_token_embeddings(len(tokenizer))

        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        if (
                hasattr(model.config, "max_position_embeddings")
                and model.config.max_position_embeddings < self.data_args.max_source_length
        ):
            if self.model_args.resize_position_embeddings is None:
                self.logger.warning(
                    f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                    f"to {self.data_args.max_source_length}."
                )
                model.resize_position_embeddings(self.data_args.max_source_length)
            elif self.model_args.resize_position_embeddings:
                model.resize_position_embeddings(self.data_args.max_source_length)
            else:
                raise ValueError(
                    f"`--max_source_length` is set to {self.data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                    f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                    "resize the model's position encodings by passing `--resize_position_embeddings`."
                )

        prefix = self.data_args.source_prefix if self.data_args.source_prefix is not None else ""

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = raw_datasets['test'].column_names
        text_column, summary_column = column_names[0], column_names[1]

        # Temporarily set max_target_length for training.
        max_target_length = self.data_args.max_target_length
        padding = self.data_args.pad_to_max_length

        if self.training_args.label_smoothing_factor > 0 and not hasattr(model,
                                                                         "prepare_decoder_input_ids_from_labels"):
            self.logger.warning(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
            )

        def preprocess_function(examples):
            # remove pairs where at least one record is None

            inputs, targets = [], []
            for i in range(len(examples[text_column])):
                if examples[text_column][i] is not None and examples[summary_column][i] is not None:
                    inputs.append(examples[text_column][i])
                    targets.append(examples[summary_column][i])

            inputs = [prefix + inp for inp in inputs]
            model_inputs = tokenizer(inputs, max_length=self.data_args.max_source_length, padding=padding,
                                     truncation=True)

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        max_target_length = self.data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if self.data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(self.data_args.max_predict_samples))
        with self.training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                # remove_columns=column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

        # Data collator
        label_pad_token_id = -100 if self.data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if self.training_args.fp16 else None,
        )

        trainer_args = {
            'model': model,
            'args': self.training_args,
            'tokenizer': tokenizer,
            'data_collator': data_collator,
            'compute_metrics': None,
            'callbacks': [EarlyStoppingCallback(early_stopping_patience=5)],
        }

        if self.training_args.dataset.startswith('combine'):
            if raw_datasets['test']['dataset_type'][0] == 'wikisql':
                dataset_type = 'wikisql'
                trainer = SubTrainer(dataset_type, **trainer_args)
            elif raw_datasets['test']['dataset_type'][0] == 'spider':
                dataset_type = 'spider'
                trainer = SubTrainer(dataset_type, **trainer_args)
            else:
                raise NotImplementedError
        elif self.training_args.dataset.startswith('spider'):
            dataset_type = 'spider'
            trainer = SubTrainer(dataset_type, **trainer_args)
        elif raw_datasets['test']['dataset_type'][0] == 'cosql':
            dataset_type = 'cosql'
            trainer = SubTrainer(dataset_type, **trainer_args)
        else:
            raise NotImplementedError

        max_length = (
            self.training_args.generation_max_length
            if self.training_args.generation_max_length is not None
            else self.data_args.val_max_target_length
        )
        num_beams = self.data_args.num_beams if self.data_args.num_beams is not None else self.training_args.generation_num_beams

        self.logger.info("*** Predict ***")
        output, spider_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )

        metrics = output.metrics
        max_predict_samples = (
            self.data_args.max_predict_samples if self.data_args.max_predict_samples is not None else len(
                predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
        predictions = tokenizer.batch_decode(
            output.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        predictions = [pred.strip() for pred in predictions]

        if stream == True:
            # save metrics & predict results
            os.makedirs(f'{self.args.output_dir}/metrics/task_{cur_task_id}/task_{to_eval_task_id}', exist_ok=True)
            output_dir = f'{self.args.output_dir}/metrics/task_{cur_task_id}/task_{to_eval_task_id}'
            output_eval_result_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_result_file, 'w', encoding='utf-8') as writer:
                writer.write("\n".join(spider_results))
            output_prediction_file = os.path.join(output_dir, "generated_predictions.txt")
            with open(output_prediction_file, "w", encoding='utf-8') as writer:
                writer.write("\n".join(predictions))
        
        else:
            output_dir = f'{self.args.output_dir}/task_{cur_task_id}'
            output_eval_result_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_result_file, 'w', encoding='utf-8') as writer:
                writer.write("\n".join(spider_results))
            output_prediction_file = os.path.join(output_dir, "generated_predictions.txt")
            with open(output_prediction_file, "w", encoding='utf-8') as writer:
                writer.write("\n".join(predictions))



def get_tasks_results(dataset, backbone_plm, out_dir, style, task_num=None):
    if "base" in backbone_plm:
        model_type = "finetune_t5-base"
    elif "large" in backbone_plm:
        model_type = "finetune_t5-large"
    else:
        raise NotImplementedError("Unknown backbone")
    #读取各个任务下的结果
    dataset_dir = f"train/data_train/spider/{dataset}"

    #1.获取当前任务下各个子任务的test数量，构建dic\
    test_size_dic = {}
    for task_id in range(task_num):
        count = 0
        task_test_pth = os.path.join(dataset_dir, f"task_{task_id}", "test_seq2seq.jsonl")
        with open(task_test_pth, 'r') as f1:
            for line in f1:
                count += 1
        test_size_dic[task_id] = count

    #2.返回EX和EM的结果矩阵  
    em_acc_array = np.zeros((task_num, task_num)) 
    ex_acc_array = np.zeros((task_num, task_num))  

    for current_id in range(task_num):
        eval_range = range(current_id + 1) if current_id == task_num - 1 else range(current_id + 2)
        for to_id in eval_range:
            with open(f"{out_dir}/{model_type}/metrics/task_{current_id}/task_{to_id}/eval_results.txt", 'r') as f2:
                for line in f2.readlines():
                    if line.startswith("count"):
                        test_num = int(line.strip()[-3:].strip())
                        assert test_num == test_size_dic[to_id]
                    if line.startswith("exact match"):
                        acc_em = float(line.strip()[-5:])
                        em_acc_array[current_id][to_id] = acc_em
                    if line.startswith("execution"):
                        acc_ex = float(line.strip()[-5:])
                        ex_acc_array[current_id][to_id] = acc_ex

    return test_size_dic, em_acc_array, ex_acc_array



def eval_stream_task(task_num: int, test_num: dict, em_acc, ex_acc):
    # 计算CL下的指标 
    acc_avg_em = [float("-inf") for i in range(task_num)]
    acc_whole_em = [float("-inf") for i in range(task_num)]
    bwt_em = [float("-inf") for i in range(task_num)]
    fwt_em = [float("-inf") for i in range(task_num)]

    acc_avg_ex = [float("-inf") for i in range(task_num)]
    acc_whole_ex = [float("-inf") for i in range(task_num)]
    bwt_ex = [float("-inf") for i in range(task_num)]
    fwt_ex = [float("-inf") for i in range(task_num)]

    for task_id in range(task_num):
        # acc_avg AND acc_whole
        acc_avg_em[task_id] =  np.mean(em_acc[task_id, 0:task_id+1])
        acc_avg_ex[task_id] =  np.mean(ex_acc[task_id, 0:task_id+1])

        acc_whole_em[task_id] = np.sum(em_acc[task_id, 0:task_id+1] * np.array(list(test_num.values())[0:task_id+1])) / np.sum(np.array(list(test_num.values())[0:task_id+1]))
        acc_whole_ex[task_id] = np.sum(ex_acc[task_id, 0:task_id+1] * np.array(list(test_num.values())[0:task_id+1])) / np.sum(np.array(list(test_num.values())[0:task_id+1]))

        # BWT AND FWT
        if task_id > 0:
            # BWT
            bwt_em_tmp = 0
            for past_id in range(task_id):
                bwt_em_tmp += em_acc[task_id][past_id] - em_acc[past_id][past_id]
            bwt_em[task_id] = bwt_em_tmp / task_id

            bwt_ex[task_id] = 0
            for past_id in range(task_id):
                bwt_ex[task_id] += ex_acc[task_id][past_id] - ex_acc[past_id][past_id]
            bwt_ex[task_id] = bwt_ex[task_id] / task_id
            
            # FWT
            fwt_em_tmp = 0
            for i in range(task_id, -1, -1):
                fwt_em_tmp += em_acc[i-1][i]
                if i-1 == 0:
                    break
            fwt_em[task_id] = fwt_em_tmp / task_id

            fwt_ex_tmp = 0
            for i in range(task_id, -1, -1):
                fwt_ex_tmp += ex_acc[i-1][i]
                if i-1 == 0:
                    break
            fwt_ex[task_id] = fwt_ex_tmp / task_id

    stream_task_result = {
        "acc_avg_em": acc_avg_em,
        "acc_whole_em": acc_whole_em, 
        "bwf_em": bwt_em,
        "fwt_em": fwt_em,
        "acc_avg_ex": acc_avg_ex, 
        "acc_whole_ex": acc_whole_ex,
        "bwf_ex": bwt_ex,
        "fwt_ex": fwt_ex
        }

    return stream_task_result


# 针对train自定义Dataset类
class MyDataset_train(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data['text'])

    def __getitem__(self, idx):
        text = self.data['text'][idx]
        sql = self.data['sql'][idx]
        db_id = self.data['db_id'][idx]
        domain_type = self.data['domain_type'][idx]
        
        model_inputs = self.tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        # Set up the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(sql, max_length=256, padding='max_length', truncation=True, return_tensors='pt')

        dataset_dict = {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "labels": labels["input_ids"].squeeze(0),
            "db_id": db_id,
            "sql": sql,
            "domain_type": domain_type
        }

        return dataset_dict


class MyDataset_others(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data['text'])

    def __getitem__(self, idx):
        text = self.data['text'][idx]
        sql = self.data['sql'][idx]
        db_id = self.data['db_id'][idx]
        
        model_inputs = self.tokenizer(text, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        # Set up the tokenizer for targets
        with self.tokenizer.as_target_tokenizer():
            label = self.tokenizer(sql, max_length=256, padding='max_length', truncation=True, return_tensors='pt')

        dataset_dict = {
            "input_ids": model_inputs["input_ids"].squeeze(0),
            "attention_mask": model_inputs["attention_mask"].squeeze(0),
            "labels": label['input_ids'].squeeze(0),
            "db_id": db_id,
            "sql": sql
        }

        return dataset_dict


class SubTrainer(Seq2SeqTrainer):
    def __init__(self, dataset_type: str, **kwargs):
        super().__init__(**kwargs)
        self.compute_metrics = self._compute_metrics
        self.output_dir = self.args.output_dir
        self.dataset_type = dataset_type
        self.spider_db_ids = ['wine_1', 'game_injury', 'assets_maintenance', 'county_public_safety', 'farm', 'wedding',
                              'architecture', 'machine_repair', 'battle_death', 'debate', 'behavior_monitoring',
                              'perpetrator', 'college_2', 'csu_1', 'hospital_1', 'college_1', 'college_3', 'flight_2',
                              'flight_4', 'ship_mission', 'ship_1', 'flight_1', 'railway', 'aircraft', 'pilot_record',
                              'flight_company', 'train_station', 'restaurants', 'yelp', 'restaurant_1', 'theme_gallery',
                              'cre_Theme_park', 'roller_coaster', 'inn_1', 'apartment_rentals', 'coffee_shop',
                              'museum_visit', 'film_rank', 'imdb', 'program_share', 'entertainment_awards', 'cinema',
                              'tvshow', 'movie_1', 'insurance_policies', 'loan_1', 'small_bank_1', 'solvency_ii',
                              'insurance_fnol', 'insurance_and_eClaims', 'real_estate_properties',
                              'tracking_software_problems', 'network_2', 'allergy_1', 'protein_institute',
                              'dog_kennels', 'network_1', 'medicine_enzyme_interaction', 'device', 'station_weather',
                              'pets_1', 'twitter_1', 'storm_record', 'browser_web', 'wta_1', 'match_season', 'wrestler',
                              'gymnast', 'bike_1', 'body_builder', 'race_track', 'formula_1', 'sports_competition',
                              'soccer_2', 'swimming', 'poker_player', 'decoration_competition', 'climbing', 'club_1',
                              'riding_club', 'university_basketball']
        if self.dataset_type == 'spider' or self.dataset_type == 'cosql':
            self.etype = 'all'
        elif self.dataset_type == 'wikisql' or 'combine_multi':
            self.etype = 'match'

    def _post_process(self, dataset: Dataset, predictions: np.ndarray):
        gold_sql = dataset['sql']
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        gold_db = dataset['db_id']
        interaction_ids = dataset['interaction_id']

        if self.dataset_type == 'cosql':
            pre_ia_id, g_ia_db = None, None
            g_ia_one, p_ia_one = [], []
            gold_interactions, pred_interactions, gold_interaction_dbs = [], [], []
            for i, ia_id in enumerate(interaction_ids):
                if ia_id != pre_ia_id and pre_ia_id is not None:
                    gold_interactions.append(g_ia_one)
                    pred_interactions.append(p_ia_one)
                    gold_interaction_dbs.append(g_ia_db)
                    g_ia_one, p_ia_one = [gold_sql[i]], [predictions[i]]
                    g_ia_db = gold_db[i]
                else:
                    g_ia_one.append(gold_sql[i])
                    p_ia_one.append(predictions[i])
                    g_ia_db = gold_db[i]
                pre_ia_id = ia_id
            gold_interactions.append(g_ia_one)
            pred_interactions.append(p_ia_one)
            gold_interaction_dbs.append(g_ia_db)

            predictions = pred_interactions
            gold_sql = gold_interactions
            gold_db = gold_interaction_dbs

        return EvalPrediction(predictions=predictions, gold_sql=gold_sql, gold_db=gold_db)

    def _compute_metrics(self, eval_prediction: EvalPrediction, etype: str, in_prediction: bool = False):
        predictions, gold_sql, gold_db = eval_prediction
        if self.dataset_type == 'combine_multi':
            if not in_prediction:
                spider_predictions, spider_gold_sql, spider_gold_db = [], [], []
                wikisql_predictions, wikisql_gold_sql, wikisql_gold_db = [], [], []
                for p, g, db in zip(predictions, gold_sql, gold_db):
                    if db in self.spider_db_ids:
                        spider_predictions.append(p)
                        spider_gold_sql.append(g)
                        spider_gold_db.append(db)
                    else:
                        wikisql_predictions.append(p)
                        wikisql_gold_sql.append(g)
                        wikisql_gold_db.append(db)
                spider_match_score, _ = spider_evaluate(gold_sql=spider_gold_sql, gold_db=spider_gold_db,
                                                        predict=spider_predictions, etype='match')
                wikisql_match_score, _ = wikisql_evaluate(gold_sql=wikisql_gold_sql, gold_db=wikisql_gold_db,
                                                          predict=wikisql_predictions, etype='match')
                match_score = (len(spider_gold_db) * spider_match_score + len(
                    wikisql_gold_db) * wikisql_match_score) / (len(spider_gold_db) + len(wikisql_gold_db))
                metrics = {'eval_exact_match': match_score}
                eval_results = None

            else:
                if gold_db[0] in self.spider_db_ids:
                    evaluation_method = spider_evaluate
                else:
                    evaluation_method = wikisql_evaluate
                match_score, eval_results = evaluation_method(gold_sql=gold_sql, gold_db=gold_db,
                                                              predict=predictions, etype='match')
                metrics = {'eval_exact_match': match_score}

        else:
            if self.dataset_type == 'spider':
                evaluation_method = spider_evaluate
            elif self.dataset_type == 'wikisql':
                evaluation_method = wikisql_evaluate
            elif self.dataset_type == 'cosql':
                evaluation_method = cosql_evaluate
            else:
                raise NotImplementedError

            if etype == 'match':
                match_score, eval_results = evaluation_method(gold_sql=gold_sql, gold_db=gold_db,
                                                              predict=predictions, etype='match')
                metrics = {'eval_exact_match': match_score}
            else:
                exec_score, match_score, eval_results = evaluation_method(gold_sql=gold_sql, gold_db=gold_db,
                                                                          predict=predictions, etype='all')
                metrics = {'eval_exec': exec_score, 'eval_exact_match': match_score}

        return metrics, eval_results

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "test",
            **gen_kwargs
    ):

        gen_kwargs = gen_kwargs.copy()
        gen_kwargs["max_length"] = 256
        self._gen_kwargs = gen_kwargs

        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        self.compute_metrics = compute_metrics

        total_batch_size = self.args.eval_batch_size * self.args.world_size

        eval_preds = self._post_process(self.eval_dataset, output.predictions)
        metrics, _ = compute_metrics(eval_preds, etype=self.etype)

        with open(f'{self.output_dir}/eval_log.jsonl', 'a', encoding='utf-8') as writer:
            writer.write(json.dumps(metrics) + '\n')

        output.metrics.update(metrics)

        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(
            self,
            test_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            **gen_kwargs
    ):
        gen_kwargs = gen_kwargs.copy()
        gen_kwargs["max_length"] = 256
        self._gen_kwargs = gen_kwargs

        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(test_dataset)
        start_time = time.time()

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Prediction",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        self.compute_metrics = compute_metrics

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        preds = self._post_process(test_dataset, output.predictions)
        pred_metric, eval_results = compute_metrics(preds, in_prediction=True, etype=self.etype)
        output.metrics.update(pred_metric)

        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output, eval_results


class SubTrainer_kl(Seq2SeqTrainer):
    def __init__(self, dataset_type: str, task_id=None, temperature=None, alpha=None, teacher_model=None, **kwargs):
        super().__init__(**kwargs)
        self.compute_metrics = self._compute_metrics
        # self.compute_loss = self._compute_loss
        self.output_dir = self.args.output_dir
        self.dataset_type = dataset_type
        self.spider_db_ids = ['wine_1', 'game_injury', 'assets_maintenance', 'county_public_safety', 'farm', 'wedding',
                              'architecture', 'machine_repair', 'battle_death', 'debate', 'behavior_monitoring',
                              'perpetrator', 'college_2', 'csu_1', 'hospital_1', 'college_1', 'college_3', 'flight_2',
                              'flight_4', 'ship_mission', 'ship_1', 'flight_1', 'railway', 'aircraft', 'pilot_record',
                              'flight_company', 'train_station', 'restaurants', 'yelp', 'restaurant_1', 'theme_gallery',
                              'cre_Theme_park', 'roller_coaster', 'inn_1', 'apartment_rentals', 'coffee_shop',
                              'museum_visit', 'film_rank', 'imdb', 'program_share', 'entertainment_awards', 'cinema',
                              'tvshow', 'movie_1', 'insurance_policies', 'loan_1', 'small_bank_1', 'solvency_ii',
                              'insurance_fnol', 'insurance_and_eClaims', 'real_estate_properties',
                              'tracking_software_problems', 'network_2', 'allergy_1', 'protein_institute',
                              'dog_kennels', 'network_1', 'medicine_enzyme_interaction', 'device', 'station_weather',
                              'pets_1', 'twitter_1', 'storm_record', 'browser_web', 'wta_1', 'match_season', 'wrestler',
                              'gymnast', 'bike_1', 'body_builder', 'race_track', 'formula_1', 'sports_competition',
                              'soccer_2', 'swimming', 'poker_player', 'decoration_competition', 'climbing', 'club_1',
                              'riding_club', 'university_basketball']
        if self.dataset_type == 'spider' or self.dataset_type == 'cosql':
            self.etype = 'all'
        elif self.dataset_type == 'wikisql' or 'combine_multi':
            self.etype = 'match'
        # teacher_model
        self.teacher_model = teacher_model
        self.task_id = task_id
        self.temperature = temperature
        self.alpha = alpha
    
    def _remove_unused_columns(self, dataset: "datasets.Dataset", description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))
        ignored_columns = list(filter(lambda x: x != 'domain_type', ignored_columns))
        
        columns = [k for k in signature_columns if k in dataset.column_names]

        return dataset.remove_columns(ignored_columns)


    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        train_dataset = self._remove_unused_columns(train_dataset, description="training")
        
        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )


    def compute_loss(self, model, inputs, return_outputs=False):
        
        domain_type = inputs.data['domain_type'].tolist()
        # 如果inputs中包含domain_type，则移除
        if 'domain_type' in inputs.data.keys():
            del inputs.data['domain_type']
            # domain_type = inputs.pop('domain_type').tolist()
        
        outputs = model(**inputs)
        loss = outputs.loss

        if self.task_id > 0 and self.teacher_model != None:
            loss_kl = torch.tensor(0.0).to(loss.device)
            with torch.no_grad():
                self.teacher_model.to(loss.device)
                teacher_outputs = self.teacher_model(**inputs)
                for i, type in enumerate(domain_type):
                    if type == 3: # 3为OOD数据
                        teacher_logits = teacher_outputs.logits
                        student_logits = outputs.logits 
                        per_loss_kl = F.kl_div(F.log_softmax(student_logits[i] / self.temperature, dim=-1), F.softmax(teacher_logits[i] / self.temperature, dim=-1), reduction='batchmean')
                        loss_kl += per_loss_kl
            
            # 按照类型个数取平均
            if domain_type.count(3) > 0:
                avg_loss_kl = loss_kl / domain_type.count(3)

            else:
                avg_loss_kl = torch.tensor(0.0).to(loss.device)

            # total_loss = (1 - self.alpha) * loss + self.alpha * loss_kl
            total_loss = (1 - self.alpha) * loss + self.alpha * avg_loss_kl
        
        else:
            total_loss = loss
            
        return total_loss
    

    def _post_process(self, dataset: Dataset, predictions: np.ndarray):
        gold_sql = dataset['sql']
        predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        gold_db = dataset['db_id']
        interaction_ids = dataset['interaction_id']

        if self.dataset_type == 'cosql':
            pre_ia_id, g_ia_db = None, None
            g_ia_one, p_ia_one = [], []
            gold_interactions, pred_interactions, gold_interaction_dbs = [], [], []
            for i, ia_id in enumerate(interaction_ids):
                if ia_id != pre_ia_id and pre_ia_id is not None:
                    gold_interactions.append(g_ia_one)
                    pred_interactions.append(p_ia_one)
                    gold_interaction_dbs.append(g_ia_db)
                    g_ia_one, p_ia_one = [gold_sql[i]], [predictions[i]]
                    g_ia_db = gold_db[i]
                else:
                    g_ia_one.append(gold_sql[i])
                    p_ia_one.append(predictions[i])
                    g_ia_db = gold_db[i]
                pre_ia_id = ia_id
            gold_interactions.append(g_ia_one)
            pred_interactions.append(p_ia_one)
            gold_interaction_dbs.append(g_ia_db)

            predictions = pred_interactions
            gold_sql = gold_interactions
            gold_db = gold_interaction_dbs

        return EvalPrediction(predictions=predictions, gold_sql=gold_sql, gold_db=gold_db)

    def _compute_metrics(self, eval_prediction: EvalPrediction, etype: str, in_prediction: bool = False):
        predictions, gold_sql, gold_db = eval_prediction
        if self.dataset_type == 'combine_multi':
            if not in_prediction:
                spider_predictions, spider_gold_sql, spider_gold_db = [], [], []
                wikisql_predictions, wikisql_gold_sql, wikisql_gold_db = [], [], []
                for p, g, db in zip(predictions, gold_sql, gold_db):
                    if db in self.spider_db_ids:
                        spider_predictions.append(p)
                        spider_gold_sql.append(g)
                        spider_gold_db.append(db)
                    else:
                        wikisql_predictions.append(p)
                        wikisql_gold_sql.append(g)
                        wikisql_gold_db.append(db)
                spider_match_score, _ = spider_evaluate(gold_sql=spider_gold_sql, gold_db=spider_gold_db,
                                                        predict=spider_predictions, etype='match')
                wikisql_match_score, _ = wikisql_evaluate(gold_sql=wikisql_gold_sql, gold_db=wikisql_gold_db,
                                                          predict=wikisql_predictions, etype='match')
                match_score = (len(spider_gold_db) * spider_match_score + len(
                    wikisql_gold_db) * wikisql_match_score) / (len(spider_gold_db) + len(wikisql_gold_db))
                metrics = {'eval_exact_match': match_score}
                eval_results = None

            else:
                if gold_db[0] in self.spider_db_ids:
                    evaluation_method = spider_evaluate
                else:
                    evaluation_method = wikisql_evaluate
                match_score, eval_results = evaluation_method(gold_sql=gold_sql, gold_db=gold_db,
                                                              predict=predictions, etype='match')
                metrics = {'eval_exact_match': match_score}

        else:
            if self.dataset_type == 'spider':
                evaluation_method = spider_evaluate
            elif self.dataset_type == 'wikisql':
                evaluation_method = wikisql_evaluate
            elif self.dataset_type == 'cosql':
                evaluation_method = cosql_evaluate
            else:
                raise NotImplementedError

            if etype == 'match':
                match_score, eval_results = evaluation_method(gold_sql=gold_sql, gold_db=gold_db,
                                                              predict=predictions, etype='match')
                metrics = {'eval_exact_match': match_score}
            else:
                exec_score, match_score, eval_results = evaluation_method(gold_sql=gold_sql, gold_db=gold_db,
                                                                          predict=predictions, etype='all')
                metrics = {'eval_exec': exec_score, 'eval_exact_match': match_score}

        return metrics, eval_results

    def evaluate(
            self,
            eval_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "test",
            **gen_kwargs
    ):

        gen_kwargs = gen_kwargs.copy()
        gen_kwargs["max_length"] = 256
        self._gen_kwargs = gen_kwargs

        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        self.compute_metrics = compute_metrics

        total_batch_size = self.args.eval_batch_size * self.args.world_size

        eval_preds = self._post_process(self.eval_dataset, output.predictions)
        metrics, _ = compute_metrics(eval_preds, etype=self.etype)

        with open(f'{self.output_dir}/eval_log.jsonl', 'a', encoding='utf-8') as writer:
            writer.write(json.dumps(metrics) + '\n')

        output.metrics.update(metrics)

        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

    def predict(
            self,
            test_dataset: Optional[Dataset] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            **gen_kwargs
    ):
        gen_kwargs = gen_kwargs.copy()
        gen_kwargs["max_length"] = 256
        self._gen_kwargs = gen_kwargs

        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(test_dataset)
        start_time = time.time()

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Prediction",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        self.compute_metrics = compute_metrics

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        preds = self._post_process(test_dataset, output.predictions)
        pred_metric, eval_results = compute_metrics(preds, in_prediction=True, etype=self.etype)
        output.metrics.update(pred_metric)

        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output, eval_results