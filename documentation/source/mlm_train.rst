Training a Masked Language Model from Scratch
=============================================

We are now ready to train our own language model from scratch. 

We run the ``scripts/run_mlm.sh`` script script to train the model. 

.. code-block:: bash
    :linenos:

    TRANSFORMERS_CACHE=/tmp/ PYTORCH_TRANSFORMERS_CACHE=/tmp/ PYTHONIOENCODING=utf-8 python src/lm/run_mlm.py \
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
    --num_train_epochs 1 \
    --overwrite_output_dir \
    --output_dir $3 \
    --report_to none \
    --cache_dir /tmp/ \
    --evaluation_strategy steps \
    --logging_steps 10000 \
    --save_steps 10000 \
    --save_total_limit 2 

However, for testing we override the parameters and write to a new script file ``scripts/run_mlm_test.sh``. 
This following argument reduces the model size to be able to train on a CPU system.

.. code-block:: bash
    :linenos:

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

We now train the MLM with the test script file and share a snapshot of the training process

.. code-block:: console
    :linenos:

    $ sh scripts/run_mlm_test.sh demo/data/lm/english_sample.txt demo/data/lm/english_sample.txt demo/model/mlm/ demo/model/tokenizer/ bert 16
    
    04/07/2022 20:12:41 - WARNING - __main__ - You are instantiating a new config instance from scratch.
    04/07/2022 20:12:41 - WARNING - __main__ - Overriding config: hidden_size=128,intermediate_size=512,num_attention_heads=4,num_hidden_layers=4,max_position_embeddings=512
    04/07/2022 20:12:41 - WARNING - __main__ - New config: BertConfig {
    "attention_probs_dropout_prob": 0.1,
    "classifier_dropout": null,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 128,
    "initializer_range": 0.02,
    "intermediate_size": 512,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 4,
    "num_hidden_layers": 4,
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
    "transformers_version": "4.14.0",
    "type_vocab_size": 2,
    "use_cache": true,
    "vocab_size": 30522
    }

    [INFO|tokenization_utils_base.py:1671] 2022-04-07 20:12:41,922 >> Didn't find file demo/model/tokenizer/vocab.json. We won't load it.
    [INFO|tokenization_utils_base.py:1671] 2022-04-07 20:12:41,922 >> Didn't find file demo/model/tokenizer/merges.txt. We won't load it.
    [INFO|tokenization_utils_base.py:1740] 2022-04-07 20:12:41,923 >> loading file None
    [INFO|tokenization_utils_base.py:1740] 2022-04-07 20:12:41,923 >> loading file None
    [INFO|tokenization_utils_base.py:1740] 2022-04-07 20:12:41,923 >> loading file demo/model/tokenizer/tokenizer.json
    [INFO|tokenization_utils_base.py:1740] 2022-04-07 20:12:41,923 >> loading file demo/model/tokenizer/added_tokens.json
    [INFO|tokenization_utils_base.py:1740] 2022-04-07 20:12:41,923 >> loading file demo/model/tokenizer/special_tokens_map.json
    [INFO|tokenization_utils_base.py:1740] 2022-04-07 20:12:41,923 >> loading file demo/model/tokenizer/tokenizer_config.json
    
    04/07/2022 20:12:42 - WARNING - __main__ - Total parameters in the model = 4.59M params
    04/07/2022 20:12:42 - WARNING - __main__ - Training new model from scratch : Total size = 4.59M params
    
    [INFO|trainer.py:1204] 2022-04-07 20:12:42,760 >> ***** Running training *****
    [INFO|trainer.py:1205] 2022-04-07 20:12:42,760 >>   Num examples = 1895
    [INFO|trainer.py:1206] 2022-04-07 20:12:42,760 >>   Num Epochs = 1
    [INFO|trainer.py:1207] 2022-04-07 20:12:42,760 >>   Instantaneous batch size per device = 8
    [INFO|trainer.py:1208] 2022-04-07 20:12:42,760 >>   Total train batch size (w. parallel, distributed & accumulation) = 8
    [INFO|trainer.py:1209] 2022-04-07 20:12:42,760 >>   Gradient Accumulation steps = 1
    [INFO|trainer.py:1210] 2022-04-07 20:12:42,760 >>   Total optimization steps = 237
    
    {'loss': 6.1333, 'learning_rate': 2.8902953586497894e-05, 'epoch': 0.42}
    {'eval_loss': 6.023196220397949, 'eval_runtime': 132.1578, 'eval_samples_per_second': 14.339, 'eval_steps_per_second': 1.793, 'epoch': 0.42}
    {'loss': 5.9755, 'learning_rate': 7.805907172995782e-06, 'epoch': 0.84}
    
    Training completed. Do not forget to share your model on huggingface.co/models =)

    {'eval_loss': 5.97206974029541, 'eval_runtime': 81.7657, 'eval_samples_per_second': 23.176, 'eval_steps_per_second': 2.899, 'epoch': 0.84}
    {'train_runtime': 533.2352, 'train_samples_per_second': 3.554, 'train_steps_per_second': 0.444, 'train_loss': 6.034984540335739, 'epoch': 1.0}
    ***** train metrics *****
    epoch                    =        1.0
    train_loss               =      6.035
    train_runtime            = 0:08:53.23
    train_samples            =       1895
    train_samples_per_second =      3.554
    train_steps_per_second   =      0.444
    04/07/2022 20:27:27 - WARNING - __main__ - *** Evaluate ***
    {'task': {'name': 'Masked Language Modeling', 'type': 'fill-mask'}}
    ***** eval metrics *****
    epoch                   =        1.0
    eval_loss               =     5.9712
    eval_runtime            = 0:01:24.94
    eval_samples            =       1895
    eval_samples_per_second =     22.308
    eval_steps_per_second   =       2.79
    perplexity              =   391.9806


The trained model is present in the following folder and ready to fine-tune

.. code-block:: console
   :linenos:

   $ ls demo/model/mlm/
   README.md
   all_results.json
   added_tokens.json
   checkpoint-100/
   checkpoint-200/
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
