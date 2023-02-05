Training a Sequence Labeler
============================

Let us now look into a short tutorial on training a sequence labeler (token classifier) using pre-trained language model.

For this tutorial, we provide a sample corpus in the folder ``demo/data/ner/en/``. The data is taken from WikiANN-NER https://huggingface.co/datasets/wikiann

.. code-block:: console
   :linenos:

    $ ls demo/data/ner/
    en

    $ ls demo/data/ner/en/
    dev.csv
    test.csv
    train.csv

The ``train``, ``dev``, and ``test`` files are in conll format. The sample snippet of the train corpus is here

.. code-block:: console
   :linenos:

    $ cat demo/data/ner/en/train.csv
    This	O
    is	O
    not	O
    Romeo	B-PER
    ,	O
    he’s	O
    some	O
    other	O
    where.	O

    Your	O
    plantain	O
    leaf	O
    is	O
    excellent	O
    for	O
    that.	O


Every word is present in it's own file followed by either a ``space`` or a ``tab`` followed by the entity label. Successive sentences are separated by an empty line.

The filenames should be the same as mentioned above

Convert CoNLL file to JSON format
*********************************

We need to convert the CoNLL file to JSON format so that we can easily load the model and perform training. We use the following script to perform the conversion.

.. code-block:: console
    :linenos:

    $ python src/tokenclassifier/helper_scripts/conll_to_json_converter.py \
        --data_dir <path to folder containing CoNLL files> \
        --column_number <column number containing the labels>

For our example, we run the following command

.. code-block:: console
    :linenos:

    $ python src/tokenclassifier/helper_scripts/conll_to_json_converter.py \
        --data_dir demo/data/ner/en/ \
        --column_number 1


Training a Token classifier
***************************

We could directly train a token classifier by specifying the hyper-parameters as follows

.. code-block:: console
    :linenos:

    $ python src/tokenclassifier/train_tc.py \
        --data <path to data folder/huggingface dataset name> \
        --model_name <model name or path> \
        --tokenizer_name <Tokenizer name or path> \
        --task_name <ner or pos> \
        --output_dir <output folder where the model will be saved> \
        --batch_size <batch size to be used> \
        --learning_rate <learning rate to be used> \
        --train_steps <maximum number of training steps> \
        --eval_steps <steps after which evaluation on dev set is performed> \
        --save_steps <steps after which the model is saved> \
        --config_name <configuration name> \
        --max_seq_len <Maximum Sequence Length after which the sequence is trimmed> \
        --perform_grid_search <Perform grid search where only the result would be stored> \
        --seed <random seed used> \
        --eval_only <Perform evaluation only>




Hyper-Parameter Tuning
**********************

We first have to select the best hyper-parameter value. For this, we monitor the loss/accuracy/f1-score on the dev set and select the best hyper-parameter. We perform a grid-search over ``batch size`` and ``learning rate`` only.

+------------------+---------------------------------------------------------------------------+
| Hyper-Parameter  | Values                                                                    |
+==================+===========================================================================+
| Batch Size       | 8, 16, 32                                                                 |
+------------------+---------------------------------------------------------------------------+
| Learning Rate    | 1e-3, 1e-4, 1e-5, 1e-6, 3e-3, 3e-4, 3e-5, 13e-6, 5e-3, 5e-4, 5e-5, 5e-6   |
+------------------+---------------------------------------------------------------------------+

We now perform hyper-parameter tuning of the sequence labeler 

.. code-block:: console
    :linenos:

    $ python src/tokenclassifier/helper_scripts/tune_hyper_parameter.py \
        --data_dir demo/data/ner/en/ \
        --configuration_name bert-custom \
        --model_name demo/model/mlm/checkpoint-200/ \
        --output_dir demo/model/ner/en/ \
        --tokenizer_name demo/model/tokenizer/ \
        --log_dir logs

The code performs hyper-parameter tuning and `Aim` library tracks the experiment in ``logs`` folder


Fine-Tuning using best Hyper-Parameter
**************************************

We now run the script ``src/tokenclassifier/helper_scripts/get_best_hyper_parameter_and_train.py`` to find the best hyper-parameter and fine-tune the model using that best hyper-parameter

..  code-block:: console
    :linenos:

    $ python src/tokenclassifier/helper_scripts/get_best_hyper_parameter_and_train.py \
        --data_dir demo/data/ner/en/ \
        --configuration_name bert-custom \
        --model_name demo/model/mlm/checkpoint-200/ \
        --output_dir demo/model/ner/en/ \
        --tokenizer_name demo/model/tokenizer/ \
        --log_dir logs

        +----+------------+-------------+----------------+
        |    |   F1-Score |   BatchSize |   LearningRate |
        +====+============+=============+================+
        |  0 |  0         |          16 |         0.001  |
        +----+------------+-------------+----------------+
        |  1 |  0.08      |          16 |         0.0001 |
        +----+------------+-------------+----------------+
        |  2 |  0.0833333 |          16 |         1e-05  |
        +----+------------+-------------+----------------+
        |  3 |  0.0833333 |          16 |         1e-06  |
        +----+------------+-------------+----------------+
        |  4 |  0         |          16 |         0.003  |
        +----+------------+-------------+----------------+
        |  5 |  0         |          16 |         0.0003 |
        +----+------------+-------------+----------------+
        |  6 |  0.0833333 |          16 |         3e-05  |
        +----+------------+-------------+----------------+
        |  7 |  0.0833333 |          16 |         3e-06  |
        +----+------------+-------------+----------------+
        |  8 |  0         |          16 |         0.005  |
        +----+------------+-------------+----------------+
        |  9 |  0         |          16 |         0.0005 |
        +----+------------+-------------+----------------+
        | 10 |  0.0833333 |          16 |         5e-05  |
        +----+------------+-------------+----------------+
        | 11 |  0.0833333 |          16 |         5e-06  |
        +----+------------+-------------+----------------+
        | 12 |  0         |          32 |         0.001  |
        +----+------------+-------------+----------------+
        | 13 |  0.08      |          32 |         0.0001 |
        +----+------------+-------------+----------------+
        | 14 |  0.0833333 |          32 |         1e-05  |
        +----+------------+-------------+----------------+
        | 15 |  0.0833333 |          32 |         1e-06  |
        +----+------------+-------------+----------------+
        | 16 |  0         |          32 |         0.003  |
        +----+------------+-------------+----------------+
        | 17 |  0         |          32 |         0.0003 |
        +----+------------+-------------+----------------+
        | 18 |  0.0833333 |          32 |         3e-05  |
        +----+------------+-------------+----------------+
        | 19 |  0.0833333 |          32 |         3e-06  |
        +----+------------+-------------+----------------+
        | 20 |  0         |          32 |         0.005  |
        +----+------------+-------------+----------------+
        | 21 |  0         |          32 |         0.0005 |
        +----+------------+-------------+----------------+
        | 22 |  0.0833333 |          32 |         5e-05  |
        +----+------------+-------------+----------------+
        | 23 |  0.0833333 |          32 |         5e-06  |
        +----+------------+-------------+----------------+
        | 24 |  0         |           8 |         0.001  |
        +----+------------+-------------+----------------+
        | 25 |  0.08      |           8 |         0.0001 |
        +----+------------+-------------+----------------+
        | 26 |  0.0833333 |           8 |         1e-05  |
        +----+------------+-------------+----------------+
        | 27 |  0.0833333 |           8 |         1e-06  |
        +----+------------+-------------+----------------+
        | 28 |  0         |           8 |         0.003  |
        +----+------------+-------------+----------------+
        | 29 |  0         |           8 |         0.0003 |
        +----+------------+-------------+----------------+
        | 30 |  0.0833333 |           8 |         3e-05  |
        +----+------------+-------------+----------------+
        | 31 |  0.0833333 |           8 |         3e-06  |
        +----+------------+-------------+----------------+
        | 32 |  0         |           8 |         0.005  |
        +----+------------+-------------+----------------+
        | 33 |  0         |           8 |         0.0005 |
        +----+------------+-------------+----------------+
        | 34 |  0.0833333 |           8 |         5e-05  |
        +----+------------+-------------+----------------+
        | 35 |  0.0833333 |           8 |         5e-06  |
        +----+------------+-------------+----------------+
        Model is demo/model/mlm/checkpoint-200/
        Best Configuration is 16 1e-05
        Best F1 is 0.08333333333333334

The command fine-tunes the model for ``5`` different random seeds. The models can be found in the folder ``demo/model/ner/en/``.

.. code-block:: console
    :linenos:

    $ ls -lh demo/model/ner/en/ | grep '^d' | awk '{print $9}
    bert-custom-model_ner_16_1e-05_4_1
    bert-custom-model_ner_16_1e-05_4_2
    bert-custom-model_ner_16_1e-05_4_3
    bert-custom-model_ner_16_1e-05_4_4
    bert-custom-model_ner_16_1e-05_4_5

The folder contains the following files

.. code-block:: console
    :linenos:

    $ ls -lh demo/model/ner/en/bert-custom-model_ner_16_1e-05_4_1/ | awk '{print $5, $9}'
    224B GOAT
    884B config.json
    417B dev_predictions.txt
    188B dev_results.txt
    3.6M pytorch_model.bin
    96B runs
    262B test_predictions.txt
    169B test_results.txt
    2.9K training_args.bin

The files ``test_predictions.txt`` and ``dev_predictions.txt`` contains the predictions from the model on ``test`` and ``dev`` set respectively.
Similarly, the files ``test_results.txt`` and ``dev_results.txt`` contains the results (F1-Score, Accuracy, etc) from the model on ``test`` and ``dev`` set respectively.

The sample snippet of the ``test_predictions.txt`` and ``dev_predictions.txt`` are presented here

.. code-block:: console
    :linenos:

    $ head demo/model/ner/en/bert-custom-model_ner_16_1e-05_4_1/test_predictions.txt 
    This O O
    is O O
    not O O
    Romeo B-PER O
    , O O
    he’s O O
    some O O
    other O O
    where. O O


The first column is the word, second column is the ground truth, and the third column is the predicted label.

.. code-block:: console
    :linenos:

    $ head demo/model/ner/en/bert-custom-model_ner_16_1e-05_4_1/test_results.txt 
    test_loss = 1.888014554977417
    test_precision = 0.0
    test_recall = 0.0
    test_f1 = 0.0
    test_runtime = 0.0331
    test_samples_per_second = 60.493
    test_steps_per_second = 30.246

The scores are bad as we have trained on a tiny corpus. Training on a larger corpus should give good results.
