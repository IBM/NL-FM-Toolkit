# 🔬 NL-FM-Toolkit
NL-FM-Toolkit stands for Natural Language - Foundation Model Toolkit. The repo contains code for work pertaining to pre-training and fine-tuning of Language Models

## 🔬 Natural Language - Foundational Model - Toolkit


[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://pages.github.com/IBM/NL-FM-Toolkit/)
![Python](https://img.shields.io/badge/python-39)
![PyTorch](https://img.shields.io/badge/pytorch-PyTorch-green)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 

The repository contains code for
- Training of language models from
    - scratch
    - existing pre-trained language models
- Fine-tuning of pre-trained language models for
    - Sequence-Labelling tasks like POS, NER, etc.
    - Text classification tasks like Sentiment Analysis, etc.


The following models are supported for training language models
- Encoder-only models (BERT like models)
    - Masked Language Modeling
    - Whole-word Masked Language Modeling
- Auto-regressive Models (GPT like models)
    - Causal Language Modeling


## 📚 Documentation

- [API docs](https://pages.github.com/IBM/NL-FM-Toolkit/modules/index.html)
- [Tutorial](https://pages.github.com/IBM/NL-FM-Toolkit/intro.html)

## ⏬ Installation

Python 3.9.0
```bash
conda create -n NLPWorkSpace python==3.9.0
conda activate NLPWorkSpace
```

Git clone the repo.

```
git clone https://github.com/IBM/NL-FM-Toolkit.git
cd NL-FM-Toolkit
```

#### Install dependencies
```bash

pip install -r requirements.txt
```

| **Task** | **ReadME** | **Tutorials** |
|:---|:---|:---|
| Tokenizer Training | [README](src/tokenizer/README.md) file present in `src/tokenizer` for more details | [Tokenizer Training Tutorial](https://pages.github.com/IBM/NL-FM-Toolkit/tokenizer_train.html) |
| Language Model | [README](src/lm/README.md) file present in `src/lm` for more details | <br> [Masked Language Model Tutorial](https://pages.github.com/IBM/NL-FM-Toolkit/mlm_train.html) </br> <br>[Causal Language Model Tutorial](https://pages.github.com/IBM/NL-FM-Toolkit/clm_train.html)  |
| Token Classification (Sequence-Labeling) Tasks | [README](src/tokenclassifier/README.md) file present in `src/token_classsifier` for more details | [Sequence Labeling Trainer](https://pages.github.com/IBM/NL-FM-Toolkit/token_classifier_train.html) |
| Sequence-Classification Tasks | [README](src/sequenceclassifier/README.md) file present in `src/sequence_classsifier` for more details | [Sequence Classification Trainer](https://pages.github.com/IBM/NL-FM-Toolkit/sequence_classifier_train.html) |


## 📁 Folder structure

The repo is organized as follows.

folder | description
:--- | :---
`src` | The core code is present in this folder.
`src/tokenizer` | Code to train a tokenizer from scratch.
`src/lm` | Code to train a language model.
`src/tokenclassifier` | Code to train a token classifier model.
`src/sequenceclassifier` | Code to train a sequence classifier model.
`src/utils` | Miscellaneous helper scripts.
`demo` | Contains the data used by the tutorial code and the folder to save the trained demo models
`examples` | The folder contains sample scripts to run the model.
`docs` | All related documentation, etc.

### To-Do Tasks

- [x] Token Classification
    - Loading Data from Huggingface Dataset

- [ ] Sequence-to-Sequence Pre-training
    - Encoder-Decoder Models (mBART, mT5 like models)
    - Denoising objective
    - Whole-word Denoising objective

- [ ] Question-Answering
- [ ] Machine Translation
