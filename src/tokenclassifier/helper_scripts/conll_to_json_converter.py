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

import datasets
from typing import List
import json
import argparse

logger = datasets.logging.get_logger(__name__)


def get_command_line_args():
    parser = argparse.ArgumentParser(description="Convert CoNLL fil to JSON")

    parser.add_argument("--data_dir", help="Path to data directory")
    parser.add_argument(
        "--column_number", default=1, type=int, help="Column number of the labels"
    )

    return parser


def _generate_examples(filepath, column_number):
    logger.info("⏳ Generating examples from = %s", filepath)
    with open(filepath, encoding="utf-8") as f:
        guid = 0
        tokens = []
        ner_tags = []

        for line in f:
            line = line.strip()
            line = line.replace(" ", "\t")

            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if len(tokens) > 0:
                    yield guid, {
                        "id": str(guid),
                        "tokens": tokens,
                        "ner_tags": ner_tags,
                    }
                    guid += 1
                    tokens = []
                    ner_tags = []
            else:
                # conll2003 tokens are space separated
                splits = line.split("\t")
                tokens.append(splits[0].strip())
                ner_tags.append(splits[column_number].strip())
        # last example
        yield guid, {
            "id": str(guid),
            "tokens": tokens,
            "ner_tags": ner_tags,
        }


def main():
    parser = get_command_line_args()
    args = parser.parse_args()

    file_splits = ["train", "dev", "test"]

    for each_file in file_splits:
        csv_file_name = os.path.join(args.data_dir, each_file + ".csv")

        json_file_name = os.path.join(args.data_dir, each_file + ".json")

        with open(json_file_name, "w", errors="ignore", encoding="utf-8") as writer:
            for each_sentence in _generate_examples(csv_file_name, args.column_number):
                writer.write(json.dumps(each_sentence[1]))
                writer.write("\n")
            writer.close()


if __name__ == "__main__":
    main()
