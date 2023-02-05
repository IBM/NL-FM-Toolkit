# 🔬 NL-FM-Toolkit
NL-FM-Toolkit stands for Natural Language - Foundation Model Toolkit. The repo contains code for work pertaining to pre-training and fine-tuning of Language Models. 

The repository was used to obtain all the results reported in the paper [**Role of Language Relatedness in Multilingual Fine-tuning of Language Models: A Case Study in Indo-Aryan Languages**](https://aclanthology.org/2021.emnlp-main.675/), Tejas Dhamecha, Rudra Murthy, Samarth Bharadwaj, Karthik Sankaranarayanan, Pushpak Bhattacharyya, EMNLP 2021

The IndoAryan Language models can be found [here](https://huggingface.co/ibm/ia-multilingual-transliterated-roberta) and [here](https://huggingface.co/ibm/ia-multilingual-original-script-roberta)

<div align="center">
    <img src="https://user-images.githubusercontent.com/13848158/155354389-d0301620-77ea-4629-a743-f7aa249e14b5.png" width="60">
    <img src="https://user-images.githubusercontent.com/13848158/155354342-7df0ef5e-63d2-4df7-b9f1-d2fc0e95f53f.png" width="60">
</div>

## 🔬 Natural Language - Foundational Model - Toolkit


[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://ibm.github.io/NL-FM-Toolkit/)
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

### What is New!
The code uses [Aim](https://github.com/aimhubio/aim) to keep track of various hyper-parameter runs and select the best hyper-parameter.


## 📚 Documentation

- [API docs](https://ibm.github.io/NL-FM-Toolkit/modules/index.html)
- [Tutorial](https://ibm.github.io/NL-FM-Toolkit/intro.html)

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
| Tokenizer Training | [README](src/tokenizer/README.md) file present in `src/tokenizer` for more details | [Tokenizer Training Tutorial](https://ibm.github.io/NL-FM-Toolkit/tokenizer_train.html) |
| Language Model | [README](src/lm/README.md) file present in `src/lm` for more details | <br> [Masked Language Model Tutorial](https://ibm.github.io/NL-FM-Toolkit/mlm_train.html) </br> <br>[Causal Language Model Tutorial](https://ibm.github.io/NL-FM-Toolkit/clm_train.html)  |
| Token Classification (Sequence-Labeling) Tasks | [README](src/tokenclassifier/README.md) file present in `src/token_classsifier` for more details | [Sequence Labeling Trainer](https://ibm.github.io/NL-FM-Toolkit/token_classifier_train.html) |
| Sequence-Classification Tasks | [README](src/sequenceclassifier/README.md) file present in `src/sequence_classsifier` for more details | [Sequence Classification Trainer](https://ibm.github.io/NL-FM-Toolkit/sequence_classifier_train.html) |


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


### Citation

If you find toolkit in your work, you can cite the paper as below:

```
@inproceedings{dhamecha-etal-2021-role,
    title = "Role of {L}anguage {R}elatedness in {M}ultilingual {F}ine-tuning of {L}anguage {M}odels: {A} {C}ase {S}tudy in {I}ndo-{A}ryan {L}anguages",
    author = "Dhamecha, Tejas  and
      Murthy, Rudra  and
      Bharadwaj, Samarth  and
      Sankaranarayanan, Karthik  and
      Bhattacharyya, Pushpak",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.675",
    doi = "10.18653/v1/2021.emnlp-main.675",
    pages = "8584--8595",
}
```
