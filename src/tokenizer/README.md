## Training Tokenizer from scratch

**Syntax:**
From the main folder of the repo:

```bash
python src/tokenizer/train_tokenizer.py --input_file <input corpus> \ 
        --name <output folder name> \ 
        --tokenizer_type <type of tokenizer to be trained> \ 
        --vocab_file <vocabulary file> --vocab_size <vocabulary size>
```

| **Parameter** | **Default** | **Options** | **Description** |
|:---|:---|---|:---|
| input_file | data/input.txt | - | path to corpus/corpora on which the tokenizer has to be trained |
| name | models/byte_tokenizer | - | path where the trained tokenizer will be saved, will be stored in models folder |
| model | byte | byte | train a byte-level tokenizer (GPT-2) |
|  |  | wordlevel | train a word-level tokenizer |
|  |  | wordpiece | train a word-piece tokenizer (BERT) |
| vocab_file | None | - | vocabulary file containing word per line which need not be split into subwords |
| vocab_size | 30000 | - | vocabulary size |
