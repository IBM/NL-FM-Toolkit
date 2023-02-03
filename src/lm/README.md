## Training Language Models from Scratch
Here we list the steps to train a language model from scratch on our corpus. We assume that you also need to train a tokenizer from scratch on your corpus.

### Training MLM from scratch

**Syntax:**
From the main folder of the repo:
```bash
sh scripts/run_mlm.sh <train file> <dev file> <output folder> <path to trained tokenizer> <model type> <number of cpu cores>
```

## Training CLM from scratch

**Syntax:**
From the main folder of the repo:
```bash
sh scripts/run_clm.sh <train file> <dev file> <output folder> <path to trained tokenizer> <model type> <number of cpu cores>
```