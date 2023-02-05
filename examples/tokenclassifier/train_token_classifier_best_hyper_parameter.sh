python src/tokenclassifier/helper_scripts/get_best_hyper_parameter_and_train.py \
    --data_dir wnut_17 \
    --configuration_name bert-custom \
    --model_name prajjwal1/bert-tiny \
    --output_dir demo/model/ner/en/ \
    --tokenizer_name prajjwal1/bert-tiny \
    --train_steps 1000 \
    --log_dir logs
