python src/tokenclassifier/helper_scripts/tune_hyper_parameter.py \
    --data_dir demo/data/ner/en/ \
    --configuration_name bert-custom \
    --model_name demo/model/mlm/checkpoint-200/ \
    --output_dir demo/model/ner/en/ \
    --tokenizer_name demo/model/tokenizer/
