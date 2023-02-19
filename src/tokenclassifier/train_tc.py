# coding=utf-8
# Copyright (c) 2022, IBM.  All rights reserved.
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

import sys
import os
import subprocess
import re
import run_tc

import argparse


def get_command_line_args():
    cli_parser = argparse.ArgumentParser(description="Train Sequence Labeler")

    cli_parser.add_argument(
        "--data", help="Path to data directory or huggingface dataset name"
    )
    cli_parser.add_argument(
        "--model_name", required=True, help="Name or path to pre-trained language model"
    )
    cli_parser.add_argument("--tokenizer_name", default=None, help="Tokenizer Name")

    cli_parser.add_argument("--output_dir", default=None, help="Output Directory")
    cli_parser.add_argument(
        "--log_dir", default=None, help="Path to folder where logs would be stored"
    )

    cli_parser.add_argument("--task_name", required=True, help="Task name")

    cli_parser.add_argument("--batch_size", default="8", help="Batch Size")
    cli_parser.add_argument("--learning_rate", default="1e-5", help="Learning Rate")

    cli_parser.add_argument("--train_steps", default="10000", help="Train Steps")
    cli_parser.add_argument("--eval_steps", default="5000", help="Eval Steps")
    cli_parser.add_argument("--save_steps", default="5000", help="Save Steps")

    cli_parser.add_argument("--config_name", default="", help="Configuration Name")

    cli_parser.add_argument(
        "--use_bilstm", default="1", help="Use Bi-LSTM layer on top of Pre-trained LM representation"
    )

    cli_parser.add_argument(
        "--max_seq_len", default="512", help="Maximum Sequence Length"
    )

    cli_parser.add_argument(
        "--perform_grid_search", default="1", help="Perform Grid Search"
    )
    cli_parser.add_argument("--seed", default="1", help="Random Seed")

    cli_parser.add_argument(
        "--eval_only", action="store_true", help="Perform Evaluation Only"
    )

    return cli_parser


def main():
    parser = get_command_line_args()

    args = parser.parse_args()

    data_dir = args.data
    model_name = args.model_name
    task_name = args.task_name

    # Batch Size
    batch_size = args.batch_size
    learning_rate = args.learning_rate

    train_steps = args.train_steps
    eval_steps = args.eval_steps
    save_steps = args.save_steps

    configuration_name = args.config_name

    max_seq_len = args.max_seq_len
    perform_grid_search = args.perform_grid_search

    seed = args.seed

    if args.tokenizer_name is not None:
        tokenizer_name = args.tokenizer_name
    else:
        tokenizer_name = args.model_name

    output_dir = args.output_dir

    output_dir_name = (
        output_dir
        + "/"
        + configuration_name
        + "_"
        + task_name
        + "_"
        + str(batch_size)
        + "_"
        + str(learning_rate)
        + "_"
        + str(train_steps)
        + "_"
        + seed
    )

    if args.log_dir is None:
        args.log_dir = "logs"

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if os.path.exists(data_dir):
        arguments = """--train_file DATADIR/train.json \
            --validation_file DATADIR/dev.json \
            --test_file DATADIR/test.json \
            --model_name_or_path MODELNAME \
            --tokenizer_name TOKENIZERNAME \
            --output_dir OUTPUTDIR \
            --max_seq_length MAXSEQLEN \
            --max_steps TRAINSTEPS \
            --save_steps SAVESTEPS \
            --eval_steps EVALSTEPS \
            --learning_rate LERNRATE \
            --per_device_train_batch_size BATCHSIZE\
            --seed RANDOM \
            --do_train \
            --do_eval \
            --do_predict \
            --report_to tensorboard \
            --task TASKNAME \
            --save_total_limit 1 \
            --overwrite_output_dir \
            --overwrite_cache \
            --early_stop \
            --log_dir LOGDIR \
            --cache_dir /tmp/
        """
    else:
        arguments = """--dataset_name DATADIR \
            --model_name_or_path MODELNAME \
            --tokenizer_name TOKENIZERNAME \
            --output_dir OUTPUTDIR \
            --max_seq_length MAXSEQLEN \
            --max_steps TRAINSTEPS \
            --save_steps SAVESTEPS \
            --eval_steps EVALSTEPS \
            --learning_rate LERNRATE \
            --per_device_train_batch_size BATCHSIZE\
            --seed RANDOM \
            --do_train \
            --do_eval \
            --do_predict \
            --report_to tensorboard \
            --task TASKNAME \
            --save_total_limit 1 \
            --overwrite_output_dir \
            --overwrite_cache \
            --early_stop \
            --log_dir LOGDIR \
            --cache_dir /tmp/
        """

    if perform_grid_search == "1":
        # Don't perform prediction on test set at the time of grid search
        arguments = arguments.replace("--do_predict ", "")
    else:
        print("Not performing grid search")

    if args.eval_only:
        arguments = arguments.replace("--do_train ", "")
        arguments = arguments.replace("--do_eval ", "")
        print("Performing evaluation only")

    if args.use_bilstm == '1':
        arguments = arguments + " --use_bilstm "

    arguments = arguments.replace("DATADIR", data_dir)
    arguments = arguments.replace("OUTPUTDIR", output_dir_name)
    arguments = arguments.replace("LOGDIR", args.log_dir)

    arguments = arguments.replace("MODELNAME", model_name)
    arguments = arguments.replace("TOKENIZERNAME", tokenizer_name)
    arguments = arguments.replace("TASKNAME", task_name)

    arguments = arguments.replace("TRAINSTEPS", train_steps)
    arguments = arguments.replace("SAVESTEPS", save_steps)
    arguments = arguments.replace("EVALSTEPS", eval_steps)

    arguments = arguments.replace("LERNRATE", learning_rate)
    arguments = arguments.replace("BATCHSIZE", batch_size)

    arguments = arguments.replace("MAXSEQLEN", max_seq_len)

    arguments = arguments.replace("RANDOM", seed)

    arguments = arguments.replace("\n", " ")
    arguments = re.sub("\s+", " ", arguments)
    arguments = arguments.strip().split(" ")

    result = run_tc.main(arguments)

    if perform_grid_search == "1":
        os.system("rm -rf " + output_dir_name)


if __name__ == "__main__":
    main()
