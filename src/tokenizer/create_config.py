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

import json
import argparse
import os

import sys

from transformers import AutoTokenizer

import codecs
import progressbar

def get_command_line_args():
    parser = argparse.ArgumentParser(description="Create config.json file")
    parser.add_argument(
        "--path",
        type=str,
        default="models/gpt2_event_tokenizer",
        help="Path where the config.json file will be created",
    )

    parser.add_argument(
        "--type",
        type=str,
        default="gpt2",
        choices=["gpt2", "mt5", "led"],
        help="Type of pre-trained model ",
    )

    parser.add_argument(
        "--vocab_size", type=int, default=30000, help="Vocabulary Size of the tokenizer"
    )

    return parser

def main(args):
    parser = get_command_line_args()
    args = parser.parse_args(args)

    
    if args.type == "gpt2":
        config = {
            "architectures": ["GPT2LMHeadModel"],
            "bos_token_id": 1,
            "decoder_start_token_id": 1,
            "eos_token_id": 2,
            "model_type": "led",
            "pad_token_id": 3,
            "torch_dtype": "float32",
            "transformers_version": "4.14.0",
            "use_cache": True,
            "vocab_size": 10104,
            "vocab_size": args.vocab_size,
        }
    elif args.type == "mt5":
        config = {
            "architectures": ["MT5ForConditionalGeneration"],
            "d_ff": 2048,
            "d_kv": 64,
            "d_model": 768,
            "decoder_start_token_id": 0,
            "dropout_rate": 0.1,
            "eos_token_id": 1,
            "feed_forward_proj": "gated-gelu",
            "initializer_factor": 1.0,
            "is_encoder_decoder": True,
            "layer_norm_epsilon": 1e-06,
            "model_type": "mt5",
            "num_decoder_layers": 6,
            "num_heads": 6,
            "num_layers": 6,
            "output_past": True,
            "pad_token_id": 0,
            "relative_attention_num_buckets": 32,
            "tie_word_embeddings": False,
            "use_cache": True,
            "vocab_size": args.vocab_size,
        }
    elif args.type == "led":
        config = {
            "activation_dropout": 0.0,
            "activation_function": "gelu",
            "architectures": ["LEDForConditionalGeneration"],
            "attention_dropout": 0.0,
            "attention_window": [512, 512, 512, 512, 512, 512],
            "bos_token_id": 0,
            "classifier_dropout": 0.0,
            "d_model": 1024,
            "decoder_attention_heads": 8,
            "decoder_ffn_dim": 2048,
            "decoder_layerdrop": 0.0,
            "decoder_layers": 6,
            "decoder_start_token_id": 2,
            "dropout": 0.1,
            "encoder_attention_heads": 8,
            "encoder_ffn_dim": 2048,
            "encoder_layerdrop": 0.0,
            "encoder_layers": 6,
            "eos_token_id": 2,
            "init_std": 0.02,
            "is_encoder_decoder": True,
            "max_decoder_position_embeddings": 1024,
            "max_encoder_position_embeddings": 16384,
            "model_type": "led",
            "num_hidden_layers": 6,
            "pad_token_id": 1,
            "torch_dtype": "float32",
            "transformers_version": "4.14.0",
            "use_cache": True,
            "vocab_size": args.vocab_size,
        }

    config["vocab_size"] = args.vocab_size
    with open(os.path.join(args.path, "config.json"), "w") as fp:
        json.dump(config, fp)


if __name__ == "__main__":
    main(sys.argv[1:])
