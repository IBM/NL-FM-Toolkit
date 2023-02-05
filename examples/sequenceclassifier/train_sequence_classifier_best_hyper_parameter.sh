python src/sequenceclassifier/helper_scripts/get_best_hyper_parameter_and_train.py \
    --data_dir rotten_tomatoes \
    --configuration_name bert-custom \
    --model_name prajjwal1/bert-tiny \
    --output_dir demo/model/sentiment/ \
    --tokenizer_name prajjwal1/bert-tiny \
    --task_name rotten_tomatoes \
    --train_steps 1000 \
    --log_dir logs
