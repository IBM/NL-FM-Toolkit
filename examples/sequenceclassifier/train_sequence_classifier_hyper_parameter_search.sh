python src/sequenceclassifier/helper_scripts/tune_hyper_parameter.py \
    --data_dir demo/data/sentiment/ \
    --configuration_name bert-custom \
    --model_name demo/model/mlm/checkpoint-200/ \
    --output_dir demo/model/sentiment/ \
    --tokenizer_name demo/model/tokenizer/ \
    --task_name sentiment
