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
Code to train a tokenizer on in-house corpus/corpora

This code takes a corpus or corpora as input and trains a sub-word tokenizer using Huggingface Transformers Library.
Optionally the code takes in a vocab file which contains words in it's own line and which shouldn't be split by the tokenizer.

"""

import argparse
import codecs
import json
import os
from pathlib import Path
import sys

import progressbar
import tokenizers
from tokenizers import AddedToken, Tokenizer, trainers
from tokenizers.decoders import BPEDecoder
from tokenizers.models import BPE, WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import (CharDelimiterSplit, Sequence, Split,
                                       WhitespaceSplit)
from tokenizers.processors import TemplateProcessing


def add_vocab_from_file(tokenizer, vocab_file):
    """
    This function reads vocabulary from the file and adds the words to the trained tokenizer
    The vocabulary file should contain every word in it's own line
    The tokenizer will not split these words

    :param tokenizer: this is the tokenizer we just trained, it could also be any pre-trained tokenizer
    :type tokenizer: AutoTokenizer

    :param vocab_file: vocabulary file containing word per line which need not be split into subwords
    :type vocab_file: str

    :return: None

    """
    vocabulary = {}

    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    with codecs.open(vocab_file, "r", errors="ignore", encoding="utf8") as f_in:
        line_number = 0
        for line in f_in:
            line_number = line_number + 1
            bar.update(line_number)

            line = line.strip()
            if line:
                for token in line.split(" "):
                    if token not in vocabulary:
                        vocabulary[token] = 1
        f_in.close()

    for token in progressbar.progressbar(vocabulary):
        tokenizer.add_tokens(AddedToken(token, single_word=True))


def get_command_line_args():
    parser = argparse.ArgumentParser(description="Train a tokenizer from scratch")

    parser.add_argument(
        "--input_file",
        type=str,
        default="data/input.txt",
        help="path to corpus/corpora on which the tokenizer has to be trained",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="models/byte_tokenizer",
        help="path where the trained tokenizer will be saved",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="byte",
        help="type of tokenizer to be trained",
        choices=["byte", "wordpiece"],
    )

    parser.add_argument(
        "--vocab_file",
        type=str,
        default=None,
        help="vocabulary file containing word per line which need not be split into subwords",
    )

    parser.add_argument("--vocab_size", type=int, default=30000, help="Vocabulary Size")

    return parser


def main(args):

    parser = get_command_line_args()

    args = parser.parse_args(args)

    tokenizer_path = args.name

    Path(args.name).mkdir(parents=True, exist_ok=True)

    input_files = args.input_file
    if "," in args.input_file:
        input_files = args.input_file.split(",")

    if args.tokenizer_type == "byte":
        # We build our custom Byte-level tokenizer:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Sequence( [WhitespaceSplit(), CharDelimiterSplit('_'), CharDelimiterSplit('/')] )
        tokenizer.normalizer = BertNormalizer(lowercase=False, clean_text=True)
        tokenizer.decoder = BPEDecoder(suffix="</w>")

        # We can train this tokenizer by giving it a list of path to text files:
        trainer = trainers.BpeTrainer(
            special_tokens=["[UNK]", "<s>", "</s>", "[PAD]", "[MASK]"],
            show_progress=True,
            vocab_size=args.vocab_size,
            end_of_word_suffix="</w>",
        )
        if isinstance(input_files, list):
            tokenizer.train(files=input_files, trainer=trainer)
        else:
            tokenizer.train(files=[input_files], trainer=trainer)

        tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> <s>:1 $B:1 </s>:1",
            special_tokens=[
                ("<s>", tokenizer.token_to_id("<s>")),
                ("</s>", tokenizer.token_to_id("</s>")),
            ],
        )

        if args.vocab_file is not None:
            add_vocab_from_file(tokenizer, args.vocab_file)

        tokenizer.save(os.path.join(tokenizer_path, "tokenizer.json"), pretty=True)

    elif args.tokenizer_type == "wordpiece":
        # We build our custom Wordpiece tokenizer:
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Sequence( [WhitespaceSplit(), CharDelimiterSplit('_'), CharDelimiterSplit('/')] )
        tokenizer.normalizer = BertNormalizer(lowercase=False, clean_text=True)
        tokenizer.decoder = tokenizers.decoders.WordPiece(prefix="</w>")

        # We can train this tokenizer by giving it a list of path to text files:
        trainer = trainers.WordPieceTrainer(
            special_tokens=["[UNK]", "<s>", "</s>", "[PAD]", "[MASK]"],
            show_progress=True,
            vocab_size=args.vocab_size,
            continuing_subword_prefix="</w>",
        )
        if isinstance(input_files, list):
            tokenizer.train(files=input_files, trainer=trainer)
        else:
            tokenizer.train(files=[input_files], trainer=trainer)

        tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> <s>:1 $B:1 </s>:1",
            special_tokens=[
                ("<s>", tokenizer.token_to_id("<s>")),
                ("</s>", tokenizer.token_to_id("</s>")),
            ],
        )

        if args.vocab_file is not None:
            add_vocab_from_file(tokenizer, args.vocab_file)

        tokenizer.save(os.path.join(tokenizer_path, "tokenizer.json"), pretty=True)


if __name__ == "__main__":
    main(sys.argv[1:])
 