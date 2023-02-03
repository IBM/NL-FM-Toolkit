# Sequence-Classification Tasks
## (Sentiment, Text Classification, ...)

The data folder containing the labelled documents should have the  following structure

```
.
|-- data_dir
|   |-- train.txt
|   |-- valid.txt
|   `-- test.txt
```

data\_dir: should contain the labelled data in TSV format with first row as header.
The input sentence should contain the header `Sentence` and output should contain the header `label`


## Performing Hyper-parameter search
Syntax:

```bash
PYTHONIOENCODING=utf-8 python src/sequenceclassifier/helper_scripts/tune_hyper_parameter.py \ 
--data_dir <data_dir> --model_name <model_name> \ 
--configuration_name <configuration name prefix for this experiment>
```

```bash
PYTHONIOENCODING=utf-8 python src/sequenceclassifier/helper_scripts/tune_hyper_parameter.py \ 
--data_dir ./data/sentiment/test_hi --model_name bert-base-multilingual-cased \ 
--configuration_name hindi_mbert_sentiment
```

This command would perform hyper-parameter grid search over `learning rate` and `batch size` and save the validation splits results to a text file in `output` folder

## Fine-tuning using the best Hyper-parameter 
Syntax:

```bash
PYTHONIOENCODING=utf-8 python src/sequenceclassifier/helper_scripts/get_best_hyper_parameter_and_train.py \ 
--data_dir <data_dir> --model_name <model_name> \ 
--configuration_name <configuration name prefix for this experiment> \ 
--filepath <path containing saved results file after running hyper-parameter search>
```

```bash
PYTHONIOENCODING=utf-8 python src/sequenceclassifier/helper_scripts/get_best_hyper_parameter_and_train.py \ 
--data_dir ./data/sentiment/test_hi --model_name bert-base-multilingual-cased \ 
--configuration_name hindi_mbert_sentiment --filepath output/
```

This command would find the hyper-parameter combination giving the best validation F-Score. The model is trained using that hyper-parameter. The command also evaluates using the `GOAT` model and the test results are stored in output folder.

