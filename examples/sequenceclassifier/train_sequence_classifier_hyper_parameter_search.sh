python src/sequenceclassifier/helper_scripts/tune_hyper_parameter.py \
    --data_dir rotten_tomatoes \
    --configuration_name bert-custom \
    --model_name prajjwal1/bert-tiny \
    --output_dir demo/model/sentiment/ \
    --tokenizer_name prajjwal1/bert-tiny \
    --task_name rotten_tomatoes \
    --train_steps 100 \
    --log_dir logs
