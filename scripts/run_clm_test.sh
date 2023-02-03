TRANSFORMERS_CACHE=/tmp/ PYTORCH_TRANSFORMERS_CACHE=/tmp/ PYTHONIOENCODING=utf-8 python src/lm/run_clm.py \
    --model_type $5 \
    --tokenizer_name $4 \
    --config_overrides="n_embd=128,n_head=4,n_layer=2,n_positions=256" \
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
    --max_seq_length 256 \
    --num_train_epochs 1 \
    --overwrite_output_dir \
    --output_dir $3 \
    --report_to none \
    --cache_dir /tmp/ \
    --evaluation_strategy steps \
    --logging_steps 10 \
    --save_steps 10 \
    --save_total_limit 2 