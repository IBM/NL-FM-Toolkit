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
""" Fine-tuning the library models for named entity recognition on CoNLL-2003 (Bert or Roberta). """

import sys
import os
import subprocess
import re
import run_seq

data_dir = sys.argv[1]
model_name = sys.argv[2]
task_name = sys.argv[3]

# Batch Size
if len(sys.argv) >= 5:
    batch_size = sys.argv[4]
else:
    batch_size = "32"

# Learning Rate
if len(sys.argv) >= 6:
    learning_rate = sys.argv[5]
else:
    learning_rate = "1e-5"

# Number of training epochs
if len(sys.argv) >= 7:
    train_epoch = sys.argv[6]
else:
    train_epoch = "3"

# configuration name
if len(sys.argv) >= 8:
    configuration_name = sys.argv[7]
else:
    configuration_name = "sentiment"

# maximum sequence length
if len(sys.argv) >= 9:
    max_seq_len = sys.argv[8]
else:
    max_seq_len = "512"

# Do hyper-parameter tuning or just training
if len(sys.argv) >= 10:
    perform_grid_search = bool(int(sys.argv[9]))
else:
    perform_grid_search = True

if len(sys.argv) >= 11:
    seed = sys.argv[10]
else:
    seed = "42"

if len(sys.argv) >= 12:
    tokenizer_name = sys.argv[11]
else:
    tokenizer_name = "bert-base-multilingual-cased"

if len(sys.argv) >= 13:
    output_dir = sys.argv[12]
else:
    output_dir = "output"

if len(sys.argv) >= 14:
    test_file = sys.argv[13]

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
    + str(train_epoch)
    + "_"
    + seed
)

arguments = """--train_file DATADIR/train.txt --validation_file DATADIR/dev.txt --test_file DATADIR/test.txt \
  --model_name_or_path MODELNAME \
  --tokenizer_name TOKENIZERNAME \
  --output_dir OUTPUTDIR \
  --overwrite_output_dir \
  --max_seq_length MAXSEQLEN \
  --num_train_epochs TRAINEPOCH \
  --learning_rate LEARNINGRATE \
  --per_device_train_batch_size BATCHSIZE \
  --save_steps 50 \
  --seed RANDOM \
  --do_train \
  --do_eval \
  --do_predict \
  --report_to tensorboard \
  --task_name TASKNAME \
  --task TASKNAME \
  --save_total_limit 1 \
  --overwrite_cache \
  --early_stop \
  --cache_dir /tmp/
"""

if perform_grid_search:
    # Don't perform prediction on test set at the time of grid search
    arguments = arguments.replace("--do_predict ", "")
else:
    print("Not performing grid search")

if len(sys.argv) >= 14:
    # perform only prediction
    arguments = arguments.replace("--do_eval", " ")
    arguments = arguments.replace("--do_train", " ")
    arguments = arguments + " --test_file " + test_file
    print("Performing only prediction")

arguments = arguments.replace("DATADIR", data_dir)
arguments = arguments.replace("OUTPUTDIR", output_dir_name)
arguments = arguments.replace("MODELNAME", model_name)
arguments = arguments.replace("TASKNAME", task_name)
arguments = arguments.replace("TRAINEPOCH", train_epoch)
arguments = arguments.replace("LEARNINGRATE", learning_rate)
arguments = arguments.replace("BATCHSIZE", batch_size)
arguments = arguments.replace("MAXSEQLEN", max_seq_len)
arguments = arguments.replace("TOKENIZERNAME", tokenizer_name)
arguments = arguments.replace("RANDOM", seed)

arguments = arguments.replace("\n", " ")
arguments = re.sub("\s+", " ", arguments)
arguments = arguments.strip().split(" ")

result = run_seq.main(arguments)

if perform_grid_search:
    model = model_name
    if "/" in model_name:
        model = model_name.split("/")[-1]

    log_file_name = (
        configuration_name
        + "_"
        + model
        + "_"
        + task_name
        + "_"
        + str(batch_size)
        + "_"
        + str(learning_rate)
        + "_results.txt"
    )
    log_file_name = os.path.join(output_dir, log_file_name)

    with open(log_file_name, "w") as writer:
        for key, value in result.items():
            writer.write("%s = %s\n" % (key, value))

    os.system("rm -rf " + output_dir_name)
