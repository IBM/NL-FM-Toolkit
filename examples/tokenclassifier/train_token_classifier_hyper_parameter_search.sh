python src/tokenclassifier/helper_scripts/tune_hyper_parameter.py \
    --data_dir wnut_17 \
    --configuration_name bert-custom \
    --model_name prajjwal1/bert-tiny \
    --output_dir demo/model/ner/en/ \
    --tokenizer_name prajjwal1/bert-tiny \
    --train_steps 100 \
    --log_dir logs
