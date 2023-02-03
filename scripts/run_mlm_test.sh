TRANSFORMERS_CACHE=/tmp/ PYTORCH_TRANSFORMERS_CACHE=/tmp/ PYTHONIOENCODING=utf-8 python src/lm/run_mlm.py \
--model_type $5 \
--tokenizer_name $4 \
--config_overrides="hidden_size=128,intermediate_size=512,num_attention_heads=4,num_hidden_layers=2,max_position_embeddings=512" \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--train_file $1 \
--validation_file $2 \
--remove_unused_columns False \
--preprocessing_num_workers $6 \
--pad_to_max_length \
--max_train_samples 100 \
--max_eval_samples 100 \
--line_by_line \
--do_train \
--do_eval \
--num_train_epochs 1 \
--overwrite_output_dir \
--output_dir $3 \
--report_to none \
--cache_dir /tmp/ \
--evaluation_strategy steps \
--logging_steps 10 \
--save_steps 10 \
--save_total_limit 2