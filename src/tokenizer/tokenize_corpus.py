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

""" 
Code to tokenize an in-house corpus/corpora using pre-trained tokenizer
"""

import os
import sys

import json
import argparse
import codecs

import transformers
from transformers import AutoTokenizer

import progressbar

import logging

logger = logging.getLogger(__name__)


def get_command_line_args():
    parser = argparse.ArgumentParser(
        description="Tokenize corpus using pre-trained tokenizer"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="path to corpus/corpora",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/model_path",
        help="path where the trained tokenizer is be saved",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="temp",
        help="output file path",
    )

    return parser


def main():
    parser = get_command_line_args()

    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = 1
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if "," in args.input_file:
        corpus = args.input_file.split(",")
    else:
        corpus = args.input_file

    tokenizer_path = args.model_path

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if isinstance(corpus, list):
        for lang_corpus in corpus:
            bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
            line_number = 0
            with codecs.open(
                lang_corpus, "r", errors="ignore", encoding="utf8"
            ) as f_in:
                with codecs.open(
                    os.path.join(args.output, lang_corpus.split("/")[-1]),
                    "w",
                    errors="ignore",
                    encoding="utf8",
                ) as f_out:
                    for line in f_in:
                        line = line.strip()

                        if line:
                            line_number += 1
                            tokenized_text = tokenizer.tokenize(line)
                            tokenized_text = [
                                word[:-4] if word.endswith("</w>") else word + "@@"
                                for word in tokenized_text
                            ]
                            f_out.write(" ".join(tokenized_text).strip() + "\n")
                            bar.update(line_number)
                    f_out.close()
                f_in.close()


if __name__ == "__main__":
    main()
