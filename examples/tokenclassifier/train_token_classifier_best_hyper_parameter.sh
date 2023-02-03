python src/tokenclassifier/helper_scripts/get_best_hyper_parameter_and_train.py \
    --data_dir demo/data/ner/en/ \
    --configuration_name bert-custom \
    --model_name demo/model/mlm/checkpoint-200/ \
    --output_dir demo/model/ner/en/ \
    --tokenizer_name demo/model/tokenizer/ \
    --filepath demo/model/ner/en/