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


def get_command_line_args():
    parser = argparse.ArgumentParser(description="Tune Best Hyper-parameter")
    parser.add_argument("--data_dir", help="Path to data directory")
    parser.add_argument("--configuration_name", help="Configuration Name")
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
    parser.add_argument(
        "--use_bilstm", default='1', help="Use Bi-LSTM layer on top of Pre-trained LM representation"
    )

    return parser


def main():
    parser = get_command_line_args()

    args = parser.parse_args()

    bs = ["8", "16", "32"]

    lr = [
        "1e-3",
        "1e-4",
        "1e-5",
        "1e-6",
        "3e-3",
        "3e-4",
        "3e-5",
        "3e-6",
        "5e-3",
        "5e-4",
        "5e-5",
        "5e-6",
    ]

    model_name = args.model_name

    num_jobs = 0
    for b in bs:
        for l in lr:
            if "/" in model_name:
                config = args.configuration_name + "-" + model_name.split("/")[1]
            else:
                config = args.configuration_name + "-" + model_name

            log_file_name = os.path.join(
                args.output_dir,
                config
                + "_"
                + model_name
                + "_ner_"
                + str(b)
                + "_"
                + str(l)
                + "_results.txt",
            )
            if os.path.exists(log_file_name):
                continue

            num_jobs = num_jobs + 1

            command = "python src/tokenclassifier/train_tc.py \
                --data {2} \
                --model_name {3} \
                --task_name ner \
                --batch_size {0} \
                --learning_rate {1} \
                --train_steps {8} \
                --config_name {4} \
                --perform_grid_search 1 \
                --seed 1 \
                --tokenizer_name {5} \
                --output_dir {6}\
                --log_dir {7}\
                --use_bilstm {9}".format(
                b,
                l,
                args.data_dir,
                model_name,
                config,
                args.tokenizer_name,
                args.output_dir,
                args.log_dir,
                args.train_steps,
                args.use_bilstm,
            )
            os.system(command)

    print("Number of jobs being executed is {0}".format(num_jobs))


if __name__ == "__main__":
    main()
