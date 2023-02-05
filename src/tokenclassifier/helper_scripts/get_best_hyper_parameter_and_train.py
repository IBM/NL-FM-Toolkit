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

import os
import argparse
import glob
import pandas as pd
from aim import Repo


def get_command_line_args():
    parser = argparse.ArgumentParser(description="Find Best Hyper-parameter")
    parser.add_argument("--data_dir", help="Path to data directory")
    parser.add_argument("--configuration_name", help="configuration_name Name")
    parser.add_argument(
        "--model_name", help="Name or path to pre-trained language model"
    )
    parser.add_argument(
        "--tokenizer_name", help="Name or path to pre-trained tokenizer"
    )
    parser.add_argument(
        "--train_steps", default=10000, help="Number of train steps to be performed"
    )
    parser.add_argument("--output_dir", help="Output Folder Name")
    parser.add_argument(
        "--log_dir", default=None, help="Path to folder where logs would be stored"
    )

    return parser


def main():
    parser = get_command_line_args()

    args = parser.parse_args()

    model_name = args.model_name

    results_dict = {}

    runs_repo = Repo(args.log_dir)

    for each_run in runs_repo.iter_runs():
        for metric in each_run.metrics():
            if "overall_f1" in metric.name and metric.context["subset"] == "eval":
                batch_size = str(each_run["hparams"]["train_batch_size"])
                learning_rate = str(each_run["hparams"]["learning_rate"])
                results_dict[
                    batch_size + " " + learning_rate
                ] = metric.values.sparse_numpy()[1][0]

    max_f1 = 0.0
    best_config = ""

    for config in results_dict:
        if results_dict[config] > max_f1:
            max_f1 = results_dict[config]
            best_config = config

    df = pd.DataFrame(
        list(results_dict.items()), columns=["BatchSize LearningRate", "F1-Score"]
    )
    df[["BatchSize", "LearningRate"]] = df["BatchSize LearningRate"].str.split(
        " ", 1, expand=True
    )
    df = df.drop("BatchSize LearningRate", axis=1)

    df["BatchSize"] = df["BatchSize"].astype(int)
    df["LearningRate"] = df["LearningRate"].astype(float)
    df["F1-Score"] = df["F1-Score"].astype(float)

    print(df.to_markdown(tablefmt="grid"))
    print("Model is {0}".format(model_name))
    print("Best Configuration is " + best_config)
    print("Best F1 is " + str(max_f1))

    parameter_string = best_config.split(" ")

    b = int(parameter_string[0])
    l = float(parameter_string[1])

    if "/" in model_name:
        config = args.configuration_name + "-" + model_name.split("/")[1]
    else:
        config = args.configuration_name + "-" + model_name

    epoch = 4

    for seed in range(1, 6):
        command = "python src/tokenclassifier/train_tc.py \
                --data {2} \
                --model_name {3} \
                --task_name ner \
                --batch_size {0} \
                --learning_rate {1} \
                --train_steps {9} \
                --config_name {4} \
                --perform_grid_search 0 \
                --seed {7} \
                --tokenizer_name {5} \
                --output_dir {6} \
                --log_dir {8}".format(
            b,
            l,
            args.data_dir,
            model_name,
            config,
            args.tokenizer_name,
            args.output_dir,
            seed,
            args.log_dir,
            args.train_steps,
        )

        os.system(command)


if __name__ == "__main__":
    main()
