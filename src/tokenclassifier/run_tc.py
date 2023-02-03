# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modifications copyright (C) 2022 IBM
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert or Roberta). """


import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import shutil
import glob

import numpy as np

from seqeval.metrics import f1_score, precision_score, recall_score
import torch
from torch import nn
from transformers import PreTrainedModel

from datasets import ClassLabel, load_dataset

import evaluate

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import (
	AutoConfig,
	AutoModelForTokenClassification,
	DataCollatorForTokenClassification,
	AutoTokenizer,
	EvalPrediction,
	HfArgumentParser,
	Trainer,
	TrainingArguments,
	set_seed,
	EarlyStoppingCallback,
	IntervalStrategy,
)
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
	"""
	Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
	"""

	model_name_or_path: str = field(
		metadata={
			"help": "Path to pretrained model or model identifier from huggingface.co/models"
		}
	)
	config_name: Optional[str] = field(
		default=None,
		metadata={
			"help": "Pretrained config name or path if not the same as model_name"
		},
	)
	tokenizer_name: Optional[str] = field(
		default=None,
		metadata={
			"help": "Pretrained tokenizer name or path if not the same as model_name"
		},
	)
	use_fast: bool = field(
		default=False, metadata={"help": "Set this flag to use fast tokenization."}
	)
	# If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
	# or just modify its tokenizer_config.json.
	cache_dir: Optional[str] = field(
		default=None,
		metadata={
			"help": "Where do you want to store the pretrained models downloaded from s3"
		},
	)
	use_auth_token: bool = field(
		default=False,
		metadata={
			"help": (
				"Will use the token generated when running `huggingface-cli login` (necessary to use this script "
				"with private models)."
			)
		},
	)


@dataclass
class DataTrainingArguments:
	"""
	Arguments pertaining to what data we are going to input our model for training and eval.
	"""

	dataset_name: Optional[str] = field(
		default=None,
		metadata={"help": "The name of the dataset to use (via the datasets library)."},
	)

	dataset_config_name: Optional[str] = field(
		default=None,
		metadata={
			"help": "The configuration name of the dataset to use (via the datasets library)."
		},
	)

	train_file: Optional[str] = field(
		default=None,
		metadata={"help": "The input training data file (a csv or JSON file)."},
	)

	validation_file: Optional[str] = field(
		default=None,
		metadata={
			"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."
		},
	)

	test_file: Optional[str] = field(
		default=None,
		metadata={
			"help": "An optional input test data file to predict on (a csv or JSON file)."
		},
	)

	text_column_name: Optional[str] = field(
		default=None,
		metadata={
			"help": "The column name of text to input in the file (a csv or JSON file)."
		},
	)

	label_column_name: Optional[str] = field(
		default=None,
		metadata={
			"help": "The column name of label to input in the file (a csv or JSON file)."
		},
	)

	max_seq_length: int = field(
		default=512,
		metadata={
			"help": "The maximum total input sequence length after tokenization. Sequences longer "
			"than this will be truncated, sequences shorter will be padded."
		},
	)
	
	pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )

	overwrite_cache: bool = field(
		default=False,
		metadata={"help": "Overwrite the cached training and evaluation sets"},
	)

	preprocessing_num_workers: Optional[int] = field(
		default=None,
		metadata={"help": "The number of processes to use for the preprocessing."},
	)

	task_name: str = field(default="ner", metadata={"help": "name of taks: pos or ner"})

	early_stop: bool = field(default=False, metadata={"help": "Use early stopping "})

	def __post_init__(self):
		if (
			self.dataset_name is None
			and self.train_file is None
			and self.validation_file is None
		):
			raise ValueError(
				"Need either a dataset name or a training/validation file."
			)
		else:
			if self.train_file is not None:
				extension = self.train_file.split(".")[-1]
				assert extension in [
					"csv",
					"json",
				], "`train_file` should be a csv or a json file."
			if self.validation_file is not None:
				extension = self.validation_file.split(".")[-1]
				assert extension in [
					"csv",
					"json",
				], "`validation_file` should be a csv or a json file."
		self.task_name = self.task_name.lower()


def main(args):
	# See all possible arguments in src/transformers/training_args.py
	# or by passing the --help flag to this script.
	# We now keep distinct sets of args, for a cleaner separation of concerns.

	parser = HfArgumentParser(
		(ModelArguments, DataTrainingArguments, TrainingArguments)
	)
	if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
		# If we pass only one argument to the script and it's the path to a json file,
		# let's parse it to get our arguments.
		model_args, data_args, training_args = parser.parse_json_file(
			json_file=os.path.abspath(sys.argv[1])
		)
	else:
		(
			model_args,
			data_args,
			training_args,
		) = parser.parse_args_into_dataclasses(args)

	if (
		os.path.exists(training_args.output_dir)
		and os.listdir(training_args.output_dir)
		and training_args.do_train
		and not training_args.overwrite_output_dir
	):
		raise ValueError(
			f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
		)

	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
	)
	logger.warning(
		"Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
		training_args.local_rank,
		training_args.device,
		training_args.n_gpu,
		bool(training_args.local_rank != -1),
		training_args.fp16,
	)
	logger.info("Training/evaluation parameters %s", training_args)

	# Detecting last checkpoint.
	last_checkpoint = None
	if (
		os.path.isdir(training_args.output_dir)
		and training_args.do_train
		and not training_args.overwrite_output_dir
	):
		last_checkpoint = get_last_checkpoint(training_args.output_dir)
		if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
			raise ValueError(
				f"Output directory ({training_args.output_dir}) already exists and is not empty. "
				"Use --overwrite_output_dir to overcome."
			)
		elif (
			last_checkpoint is not None and training_args.resume_from_checkpoint is None
		):
			logger.info(
				f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
				"the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
			)

	# Set seed
	set_seed(training_args.seed)

	if data_args.dataset_name is not None:
		# Downloading and loading a dataset from the hub.
		raw_datasets = load_dataset(
			data_args.dataset_name,
			data_args.dataset_config_name,
			cache_dir=model_args.cache_dir,
			use_auth_token=True if model_args.use_auth_token else None,
		)
	else:
		data_files = {}
		if data_args.train_file is not None:
			data_files["train"] = data_args.train_file
		if data_args.validation_file is not None:
			data_files["validation"] = data_args.validation_file
		if data_args.test_file is not None:
			data_files["test"] = data_args.test_file
		extension = data_args.train_file.split(".")[-1]
		raw_datasets = load_dataset(
			extension,
			data_files=data_files,
			cache_dir=model_args.cache_dir,
		)
		
	# Prepare CONLL-2003 task
	if training_args.do_train:
		column_names = raw_datasets["train"].column_names
		features = raw_datasets["train"].features
	else:
		column_names = raw_datasets["validation"].column_names
		features = raw_datasets["validation"].features

	if data_args.text_column_name is not None:
		text_column_name = data_args.text_column_name
	elif "tokens" in column_names:
		text_column_name = "tokens"
	else:
		text_column_name = column_names[0]

	if data_args.label_column_name is not None:
		label_column_name = data_args.label_column_name
	elif f"{data_args.task_name}_tags" in column_names:
		label_column_name = f"{data_args.task_name}_tags"
	else:
		label_column_name = column_names[1]

	# In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
	# unique labels.
	def get_label_list(labels):
		unique_labels = set()
		for label in labels:
			unique_labels = unique_labels | set(label)
		label_list = list(unique_labels)
		label_list.sort()
		return label_list

	# If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
	# Otherwise, we have to get the list of labels manually.
	labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
	if labels_are_int:
		label_list = features[label_column_name].feature.names
		label_to_id = {i: i for i in range(len(label_list))}
		id_to_label = {i: l for i, l in enumerate(label_list)}
	else:
		label_list = get_label_list(raw_datasets["train"][label_column_name])
		label_to_id = {l: i for i, l in enumerate(label_list)}
		id_to_label = {i: l for i, l in enumerate(label_list)}

	num_labels = len(label_list)

	logger.warning("Labels to ID are {0}".format(label_to_id))
	logger.warning("ID to Labels are {0}".format(id_to_label))

	# Load pretrained model and tokenizer
	#
	# Distributed training:
	# The .from_pretrained methods guarantee that only one local process can concurrently
	# download model & vocab.

	config = AutoConfig.from_pretrained(
		model_args.config_name
		if model_args.config_name
		else model_args.model_name_or_path,
		num_labels=num_labels,
		id2label=id_to_label,
		label2id=label_to_id,
		cache_dir=model_args.cache_dir,
	)
	tokenizer = AutoTokenizer.from_pretrained(
		model_args.tokenizer_name
		if model_args.tokenizer_name
		else model_args.model_name_or_path,
		cache_dir=model_args.cache_dir,
		use_fast="True",
	)
	model = AutoModelForTokenClassification.from_pretrained(
		model_args.model_name_or_path,
		from_tf=bool(".ckpt" in model_args.model_name_or_path),
		config=config,
		cache_dir=model_args.cache_dir,
	)

	n_params = 0
	for name, p in model.named_parameters():
		n_params = n_params + p.numel()

	logger.warning( f"Fine-Tuning existing model - Total size = {n_params/2**20:.2f}M params" )

	# If we are not training then use labels from models
	if not training_args.do_train:
		if list(sorted(model.config.label2id.keys())) == list(sorted(label_list)):
			# Reorganize `label_list` to match the ordering of the model.
			if labels_are_int:
				label_to_id = {
					i: int(model.config.label2id[l]) for i, l in enumerate(label_list)
				}
				label_list = [model.config.id2label[i] for i in range(num_labels)]
			else:
				label_list = [model.config.id2label[i] for i in range(num_labels)]
				label_to_id = {l: i for i, l in enumerate(label_list)}
		else:
			logger.warning(
				"Your model seems to have been trained with labels, but they don't match the dataset: ",
				f"model labels: {list(sorted(model.config.label2id.keys()))}, dataset labels:"
				f" {list(sorted(label_list))}.\nIgnoring the model labels as a result.",
			)

	# Set the correspondences label/ID inside the model config
	model.config.label2id = {l: i for i, l in enumerate(label_list)}
	model.config.id2label = {i: l for i, l in enumerate(label_list)}

	# Map that sends B-Xxx label to its I-Xxx counterpart
	b_to_i_label = []
	for idx, label in enumerate(label_list):
		if label.startswith("B-") and label.replace("B-", "I-") in label_list:
			b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
		else:
			b_to_i_label.append(idx)

	padding = "max_length" if data_args.pad_to_max_length else False

	# Tokenize all texts and align the labels with them.
	def tokenize_and_align_labels(examples):
		tokenized_inputs = tokenizer(
			examples[text_column_name],
			padding=padding,
			truncation=True,
			max_length=data_args.max_seq_length,
			# We use this argument because the texts in our dataset are lists of words (with a label for each word).
			is_split_into_words=True,
		)
		labels = []
		for i, label in enumerate(examples[label_column_name]):
			word_ids = tokenized_inputs.word_ids(batch_index=i)
			previous_word_idx = None
			label_ids = []
			for word_idx in word_ids:
				# Special tokens have a word id that is None. We set the label to -100 so they are automatically
				# ignored in the loss function.
				if word_idx is None:
					label_ids.append(-100)
				# We set the label for the first token of each word.
				elif word_idx != previous_word_idx:
					label_ids.append(label_to_id[label[word_idx]])
				# For the other tokens in a word, we set the label to either the current label or -100, depending on
				# the label_all_tokens flag.
				else:
					label_ids.append(-100)
				previous_word_idx = word_idx

			labels.append(label_ids)
		tokenized_inputs["labels"] = labels
		return tokenized_inputs

	if training_args.do_train:
		if "train" not in raw_datasets:
			raise ValueError("--do_train requires a train dataset")
		train_dataset = raw_datasets["train"]

		with training_args.main_process_first(desc="train dataset map pre-processing"):
			train_dataset = train_dataset.map(
				tokenize_and_align_labels,
				batched=True,
				num_proc=data_args.preprocessing_num_workers,
				load_from_cache_file=not data_args.overwrite_cache,
				desc="Running tokenizer on train dataset",
			)

	if training_args.do_eval:
		if "validation" not in raw_datasets:
			raise ValueError("--do_eval requires a validation dataset")
		eval_dataset = raw_datasets["validation"]

		with training_args.main_process_first(
			desc="validation dataset map pre-processing"
		):
			eval_dataset = eval_dataset.map(
				tokenize_and_align_labels,
				batched=True,
				num_proc=data_args.preprocessing_num_workers,
				load_from_cache_file=not data_args.overwrite_cache,
				desc="Running tokenizer on validation dataset",
			)

	if training_args.do_predict:
		if "test" not in raw_datasets:
			raise ValueError("--do_predict requires a test dataset")
		predict_dataset = raw_datasets["test"]

		with training_args.main_process_first(
			desc="prediction dataset map pre-processing"
		):
			predict_dataset = predict_dataset.map(
				tokenize_and_align_labels,
				batched=True,
				num_proc=data_args.preprocessing_num_workers,
				load_from_cache_file=not data_args.overwrite_cache,
				desc="Running tokenizer on prediction dataset",
			)

	# Data collator
	data_collator = DataCollatorForTokenClassification(
		tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
	)

	# Metrics
	metric = evaluate.load("seqeval")

	def compute_metrics(p):
		predictions, labels = p
		predictions = np.argmax(predictions, axis=2)

		# Remove ignored index (special tokens)
		true_predictions = [
			[label_list[p] for (p, l) in zip(prediction, label) if l != -100]
			for prediction, label in zip(predictions, labels)
		]
		true_labels = [
			[label_list[l] for (p, l) in zip(prediction, label) if l != -100]
			for prediction, label in zip(predictions, labels)
		]

		results = metric.compute(predictions=true_predictions, references=true_labels)
		# Unpack nested dictionaries
		final_results = {}
		for key, value in results.items():
			if isinstance(value, dict):
				for n, v in value.items():
					final_results[f"{key}_{n}"] = v
			else:
				final_results[key] = value
		return final_results

	# Initialize our Trainer
	logger.warning("Initializing Trainer")
	early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)
	training_args.metric_for_best_model = "eval_overall_f1"
	training_args.load_best_model_at_end = True
	training_args.evaluation_strategy = IntervalStrategy.STEPS
	training_args.eval_steps = training_args.save_steps
	training_args.greater_is_better = True

	# Initialize our Trainer
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		compute_metrics=compute_metrics,
		data_collator=data_collator,
		callbacks=[early_stopping_callback] if data_args.early_stop else None,
	)

	# Training
	if training_args.do_train:
		checkpoint = None
		if training_args.resume_from_checkpoint is not None:
			checkpoint = training_args.resume_from_checkpoint
		elif last_checkpoint is not None:
			checkpoint = last_checkpoint
		train_result = trainer.train(resume_from_checkpoint=checkpoint)

		metrics = train_result.metrics

		trainer.save_model()
		tokenizer.save_pretrained(training_args.output_dir)

		trainer.log_metrics("train", metrics)
		trainer.save_metrics("train", metrics)
		trainer.save_state()

	# Perform Evaluation on dev data
	results = {}
	if training_args.do_eval:
		logger.info("*** Evaluate ***")

		metrics = trainer.evaluate()

		trainer.log_metrics("eval", metrics)
		trainer.save_metrics("eval", metrics)

	# Predict
	if training_args.do_predict:
		# test on user-specified test file and not standard test file

		logger.info("*** Predict ***")

		predictions, labels, metrics = trainer.predict(
			predict_dataset, metric_key_prefix="predict"
		)
		predictions = np.argmax(predictions, axis=2)

		# Remove ignored index (special tokens)
		true_predictions = [
			[label_list[p] for (p, l) in zip(prediction, label) if l != -100]
			for prediction, label in zip(predictions, labels)
		]

		trainer.log_metrics("predict", metrics)
		trainer.save_metrics("predict", metrics)

		# Save predictions
		output_predictions_file = os.path.join(
			training_args.output_dir, "predictions.txt"
		)
		if trainer.is_world_process_zero():
			with open(output_predictions_file, "w") as writer:
				for prediction in true_predictions:
					writer.write(" ".join(prediction) + "\n")

	return metrics


def _mp_fn(index):
	# For xla_spawn (TPUs)
	main()


if __name__ == "__main__":
	main(sys.argv[1:])
