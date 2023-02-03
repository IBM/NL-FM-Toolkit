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
import sys
import codecs
import argparse
import numpy as np
import glob
import codecs
import pandas as pd
import matplotlib.pyplot as plt

from seqeval.metrics import classification_report
from seqeval.metrics import f1_score
from seqeval.scheme import IOB2


def get_command_line_args():
    parser = argparse.ArgumentParser(description="Get F-Score fomr predictions file")
    parser.add_argument("--filepath", help="Path to predictions file")

    return parser


def main():
    parser = get_command_line_args()

    args = parser.parse_args()

    document_predictions = []
    document_gold = []

    predictions = []
    gold_label = []

    with codecs.open(
        os.path.join(args.filepath),
        "r",
        errors="ignore",
        encoding="utf8",
    ) as reader:
        for line in reader:
            if "\t" not in line:
                line = line.replace(" ", "\t")

            if line == "\n":
                if len(predictions) > 0:
                    document_predictions.append(predictions)
                    document_gold.append(gold_label)
                predictions = []
                gold_label = []
            else:
                line = line.strip()

				# last but one column is the gold label
				# last column is the predicted label
                predictions.append(line.split("\t")[-1])
                gold_label.append(line.split("\t")[-2])

        reader.close()

        connl_score = f1_score(
            document_gold, document_predictions, mode="strict", scheme=IOB2
        )
        report = classification_report(
            document_gold,
            document_predictions,
            mode="strict",
            scheme=IOB2,
            digits=4,
            output_dict=True,
        )
        report["conll"] = {}
        report["conll"]["f1-score"] = connl_score

    print(report)


if __name__ == "__main__":
    main()
