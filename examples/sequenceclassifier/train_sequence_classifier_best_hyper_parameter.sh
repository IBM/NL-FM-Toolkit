python src/sequenceclassifier/helper_scripts/get_best_hyper_parameter_and_train.py \
    --data_dir demo/data/sentiment/ \
    --configuration_name bert-custom \
    --model_name demo/model/mlm/checkpoint-200/ \
    --output_dir demo/model/sentiment/ \
    --tokenizer_name demo/model/tokenizer/ \
    --filepath demo/model/sentiment/