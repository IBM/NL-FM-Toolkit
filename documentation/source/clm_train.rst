Training a Causal Language Model from Scratch
=============================================

We are now ready to train our own language model from scratch. 

We run the ``scripts/run_clm.sh`` script script to train the model. 

.. code-block:: bash
    :linenos:

    TRANSFORMERS_CACHE=/tmp/ PYTORCH_TRANSFORMERS_CACHE=/tmp/ PYTHONIOENCODING=utf-8 python src/lm/run_clm.py \
    --model_type $5 \
    --tokenizer_name $4 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --train_file $1 \
    --validation_file $2 \
    --remove_unused_columns False \
    --preprocessing_num_workers $6 \
    --pad_to_max_length \
    --line_by_line \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --num_train_epochs 1 \
    --overwrite_output_dir \
    --output_dir $3 \
    --report_to none \
    --cache_dir /tmp/ \
    --evaluation_strategy steps \
    --logging_steps 10000 \
    --save_steps 10000 \
    --save_total_limit 2 

However, for testing we override the parameters and write to a new script file ``scripts/run_clm_test.sh``. This following argument reduces the model size to be able to train on a CPU system.

.. code-block:: bash
    :linenos:

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

We now train the CLM with the test script file and share a snapshot of the training process

.. code-block:: console
    :linenos:

    $ sh scripts/run_clm_test.sh demo/data/lm/english_sample.txt demo/data/lm/english_sample.txt demo/model/clm/ demo/model/tokenizer/ gpt2 16
    
    04/07/2022 21:16:29 - WARNING - __main__ - You are instantiating a new config instance from scratch.
    04/07/2022 21:16:29 - INFO - __main__ - Overriding config: n_embd=128,n_head=4,n_layer=4,n_positions=256

    [INFO|tokenization_utils_base.py:1671] 2022-04-07 21:16:29,818 >> Didn't find file demo/model/tokenizer/vocab.json. We won't load it.
    [INFO|tokenization_utils_base.py:1671] 2022-04-07 21:16:29,818 >> Didn't find file demo/model/tokenizer/merges.txt. We won't load it.
    [INFO|tokenization_utils_base.py:1740] 2022-04-07 21:16:29,818 >> loading file None
    [INFO|tokenization_utils_base.py:1740] 2022-04-07 21:16:29,818 >> loading file None
    [INFO|tokenization_utils_base.py:1740] 2022-04-07 21:16:29,818 >> loading file demo/model/tokenizer/tokenizer.json
    [INFO|tokenization_utils_base.py:1740] 2022-04-07 21:16:29,818 >> loading file demo/model/tokenizer/added_tokens.json
    [INFO|tokenization_utils_base.py:1740] 2022-04-07 21:16:29,818 >> loading file demo/model/tokenizer/special_tokens_map.json
    [INFO|tokenization_utils_base.py:1740] 2022-04-07 21:16:29,818 >> loading file demo/model/tokenizer/tokenizer_config.json

    04/07/2022 21:16:30 - INFO - __main__ - Training new model from scratch - Total size=6.92M params
    
    [INFO|trainer.py:1204] 2022-04-07 20:12:42,760 >> ***** Running training *****
    [INFO|trainer.py:1205] 2022-04-07 20:12:42,760 >>   Num examples = 1895
    [INFO|trainer.py:1206] 2022-04-07 20:12:42,760 >>   Num Epochs = 1
    [INFO|trainer.py:1207] 2022-04-07 20:12:42,760 >>   Instantaneous batch size per device = 8
    [INFO|trainer.py:1208] 2022-04-07 20:12:42,760 >>   Total train batch size (w. parallel, distributed & accumulation) = 8
    [INFO|trainer.py:1209] 2022-04-07 20:12:42,760 >>   Gradient Accumulation steps = 1
    [INFO|trainer.py:1210] 2022-04-07 20:12:42,760 >>   Total optimization steps = 237
    
    {'loss': 5.9329, 'learning_rate': 2.8902953586497894e-05, 'epoch': 0.42}
    {'eval_loss': 5.720452785491943, 'eval_runtime': 30.5425, 'eval_samples_per_second': 62.045, 'eval_steps_per_second': 7.76, 'epoch': 0.42}

    {'loss': 5.6865, 'learning_rate': 7.805907172995782e-06, 'epoch': 0.84}
    {'eval_loss': 5.609338760375977, 'eval_runtime': 30.8089, 'eval_samples_per_second': 61.508, 'eval_steps_per_second': 7.693, 'epoch': 0.84}

    Training completed. Do not forget to share your model on huggingface.co/models =)

    {'train_runtime': 220.6908, 'train_samples_per_second': 8.587, 'train_steps_per_second': 1.074, 'train_loss': 5.776248851405921, 'epoch': 1.0}

    ***** eval metrics *****
    epoch                   =        1.0
    eval_loss               =     5.6093
    eval_runtime            = 0:00:36.93
    eval_samples            =       1895
    eval_samples_per_second =     51.301
    eval_steps_per_second   =      6.416
    perplexity              =   272.9637
 
    [INFO|modelcard.py:456] 2022-04-07 21:28:38,572 >> Dropping the following result as it does not have all the necessary fields:
 
    {'task': {'name': 'Causal Language Modeling', 'type': 'text-generation'}}

The trained model is present in the following folder and ready to fine-tune

.. code-block:: console
   :linenos:

   $ ls demo/model/clm/
   README.md
   added_tokens.json
   all_results.json
   checkpoint-200/
   checkpoint-100/
   config.json
   eval_results.json	
   pytorch_model.bin
   special_tokens_map.json
   tokenizer_config.json
   tokenizer.json
   trainer_state.json
   train_results.json
   training_args.bin
   vocab.txt

