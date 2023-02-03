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

def get_command_line_args():
    parser = argparse.ArgumentParser(description="Find Best Hyper-parameter")
    parser.add_argument(
        "--filepath", help="Path containing best hyper-parameter results"
    )
    parser.add_argument("--data_dir", help="Path to data directory")
    parser.add_argument("--configuration_name", help="configuration_name Name")
    parser.add_argument(
        "--model_name", help="Name or path to pre-trained language model"
    )
    parser.add_argument(
        "--tokenizer_name", help="Name or path to pre-trained tokenizer"
    )
    parser.add_argument("--output_dir", help="Output Folder Name")
    parser.add_argument(
        "--task_name",
        type=str,
        default="sentiment",
        help="The sequence classification task to be performed",
        choices=[
            "cola",
            "mnli",
            "mrpc",
            "qnli",
            "qqp",
            "rte",
            "sst2",
            "stsb",
            "wnli",
            "sentiment",
        ],
    )

    return parser

def main():
    parser = get_command_line_args()
    
    args = parser.parse_args()

    results_files = list(sorted(glob.glob(args.filepath + "*txt")))

    model = args.model_name

    if "/" in model:
        config = args.configuration_name + "-" + model.split("/")[1]
    else:
        config = args.configuration_name + "-" + model

    results_dict = {}

    for every_file in results_files:
        with open(every_file, "r", errors="ignore", encoding="utf8") as f_in:
            seen = False

            if config not in every_file:
                continue

            parameter_string = (
                every_file.split(args.filepath)[1].split("_results.txt")[0].split("_")
            )
            b = int(parameter_string[-2])
            l = float(parameter_string[-1])

            for line in f_in:
                line = line.strip()

                if line.startswith("eval_f1"):
                    seen = True
                    f1 = float(line.split("eval_f1 = ")[1].strip())
                    results_dict[str(b) + " " + str(l)] = f1
            if not seen:
                results_dict[str(b) + " " + str(l)] = 0.0
            f_in.close()

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
    print("Model is {0}".format(model))
    print("Best Configuration is " + best_config)
    print("Best F1 is " + str(max_f1))

    parameter_string = best_config.split(" ")

    b = int(parameter_string[0])
    l = float(parameter_string[1])

    if "/" in model:
        config = args.configuration_name + "-" + args.model_name.split("/")[1]
    else:
        config = args.configuration_name + "-" + args.model_name

    epoch = 4
    
    for seed in range(1, 6):
        command = "python src/sequenceclassifier/train_seq.py {0} {1} {2} {3} {4} {5} {6} {7} 0 {8} {9} {10}".format(
            args.data_dir,
            model,
            args.task_name,
            b,
            l,
            epoch,
            config,
            256,
            seed,
            args.tokenizer_name,
            args.output_dir
        )
        os.system(command)


if __name__ == "__main__":
    main()
