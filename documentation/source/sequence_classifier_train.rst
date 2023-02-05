Training a Sequence Classifier
==============================

Let us now look into a short tutorial on training a sequence classifier using pre-trained language model.

For this tutorial, we provide a sample corpus in the folder ``demo/data/sentiment/``. 

.. code-block:: console
   :linenos:

    $ ls demo/data/sentiment/
    dev.txt
    test.txt
    train.txt

The ``train``, ``dev``, and ``test`` files are in tab separated format. The sample snippet of the train corpus is here. The first line of the file should contain `sentence` as the name of first column and `Label` as the name of the second column (which is also the column containing class labels)

.. code-block:: console
   :linenos:

    $ cat demo/data/sentiment/train.txt
    sentence	Label
    I liked the movie	1
    I hated the movie	0
    The movie was good	1

The filenames should be the same as mentioned above

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

We now perform hyper-parameter tuning of the sequence classifier

.. code-block:: console
    :linenos:

    $ python src/sequenceclassifier/helper_scripts/tune_hyper_parameter.py \
        --data_dir demo/data/sentiment/ \
        --configuration_name bert-custom \
        --model_name demo/model/mlm/checkpoint-200/ \
        --output_dir demo/model/sentiment/ \
        --tokenizer_name demo/model/tokenizer/ \
        --task_name sentiment \
        --log_dir logs

The code performs hyper-parameter tuning and `Aim` library tracks the experiment in ``logs`` folder


Fine-Tuning using best Hyper-Parameter
**************************************

We now run the script ``src/sequenceclassifier/helper_scripts/get_best_hyper_parameter_and_train.py`` to find the best hyper-parameter and fine-tune the model using that best hyper-parameter

..  code-block:: console
    :linenos:

    $ python src/sequenceclassifier/helper_scripts/get_best_hyper_parameter_and_train.py \
        --data_dir demo/data/sentiment/ \
        --configuration_name bert-custom \
        --model_name demo/model/mlm/checkpoint-200/ \
        --output_dir demo/model/sentiment/ \
        --tokenizer_name demo/model/tokenizer/ \
        --log_dir logs

        +----+------------+-------------+----------------+
        |    |   F1-Score |   BatchSize |   LearningRate |
        +====+============+=============+================+
        |  0 |   0.666667 |          16 |         0.001  |
        +----+------------+-------------+----------------+
        |  1 |   0.666667 |          16 |         0.0001 |
        +----+------------+-------------+----------------+
        |  2 |   0        |          16 |         1e-05  |
        +----+------------+-------------+----------------+
        |  3 |   0        |          16 |         1e-06  |
        +----+------------+-------------+----------------+
        |  4 |   0.666667 |          16 |         0.003  |
        +----+------------+-------------+----------------+
        |  5 |   0.666667 |          16 |         0.0003 |
        +----+------------+-------------+----------------+
        |  6 |   0        |          16 |         3e-05  |
        +----+------------+-------------+----------------+
        |  7 |   0        |          16 |         3e-06  |
        +----+------------+-------------+----------------+
        |  8 |   0        |          16 |         0.005  |
        +----+------------+-------------+----------------+
        |  9 |   0.666667 |          16 |         0.0005 |
        +----+------------+-------------+----------------+
        | 10 |   0        |          16 |         5e-05  |
        +----+------------+-------------+----------------+
        | 11 |   0        |          16 |         5e-06  |
        +----+------------+-------------+----------------+
        | 12 |   0.666667 |          32 |         0.001  |
        +----+------------+-------------+----------------+
        | 13 |   0.666667 |          32 |         0.0001 |
        +----+------------+-------------+----------------+
        | 14 |   0        |          32 |         1e-05  |
        +----+------------+-------------+----------------+
        | 15 |   0        |          32 |         1e-06  |
        +----+------------+-------------+----------------+
        | 16 |   0.666667 |          32 |         0.003  |
        +----+------------+-------------+----------------+
        | 17 |   0.666667 |          32 |         0.0003 |
        +----+------------+-------------+----------------+
        | 18 |   0        |          32 |         3e-05  |
        +----+------------+-------------+----------------+
        | 19 |   0        |          32 |         3e-06  |
        +----+------------+-------------+----------------+
        | 20 |   0        |          32 |         0.005  |
        +----+------------+-------------+----------------+
        | 21 |   0.666667 |          32 |         0.0005 |
        +----+------------+-------------+----------------+
        | 22 |   0        |          32 |         5e-05  |
        +----+------------+-------------+----------------+
        | 23 |   0        |          32 |         5e-06  |
        +----+------------+-------------+----------------+
        | 24 |   0.666667 |           8 |         0.001  |
        +----+------------+-------------+----------------+
        | 25 |   0.666667 |           8 |         0.0001 |
        +----+------------+-------------+----------------+
        | 26 |   0        |           8 |         1e-05  |
        +----+------------+-------------+----------------+
        | 27 |   0        |           8 |         1e-06  |
        +----+------------+-------------+----------------+
        | 28 |   0.666667 |           8 |         0.003  |
        +----+------------+-------------+----------------+
        | 29 |   0.666667 |           8 |         0.0003 |
        +----+------------+-------------+----------------+
        | 30 |   0        |           8 |         3e-05  |
        +----+------------+-------------+----------------+
        | 31 |   0        |           8 |         3e-06  |
        +----+------------+-------------+----------------+
        | 32 |   0        |           8 |         0.005  |
        +----+------------+-------------+----------------+
        | 33 |   0.666667 |           8 |         0.0005 |
        +----+------------+-------------+----------------+
        | 34 |   0        |           8 |         5e-05  |
        +----+------------+-------------+----------------+
        | 35 |   0        |           8 |         5e-06  |
        +----+------------+-------------+----------------+
        Model is demo/model/mlm/checkpoint-200/
        Best Configuration is 16 0.001
        Best F1 is 0.6666666666666666

The command fine-tunes the model for ``5`` different random seeds. The models can be found in the folder ``demo/model/sentiment/``

.. code-block:: console
    :linenos:

    $ ls -lh demo/model/sentiment/ | grep '^d' | awk '{print $9}
    bert-custom-model_sentiment_16_0.001_4_1
    bert-custom-model_sentiment_16_0.001_4_2
    bert-custom-model_sentiment_16_0.001_4_3
    bert-custom-model_sentiment_16_0.001_4_4
    bert-custom-model_sentiment_16_0.001_4_5

The folder contains the following files

.. code-block:: console
    :linenos:

    $ ls -lh demo/model/sentiment/bert-custom-model_sentiment_16_0.001_4_1/ | awk '{print $5, $9}'
    386B all_results.json
    700B config.json
    219B eval_results.json
    41B predict_results_sentiment.txt
    3.6M pytorch_model.bin
    96B runs
    48B test_predictions.txt
    147B test_results.json
    187B train_results.json
    808B trainer_state.json
    2.9K training_args.bin

The files ``test_predictions.txt`` contains the predictions from the model on ``test`` set.
Similarly, the files ``test_results.json`` and ``eval_results.json`` contains the results (F1-Score, Accuracy, etc) from the model on ``test`` and ``dev`` set respectively.

The sample snippet of the ``eval_results.jsom`` is presented here

.. code-block:: console
    :linenos:

    $ head demo/model/ner/en/bert-custom-model_ner_16_1e-05_4_1/eval_results.json
    {
    "epoch": 4.0,
    "eval_f1": 0.6666666666666666,
    "eval_loss": 0.7115099430084229,
    "eval_runtime": 0.0788,
    "eval_samples": 6,
    "eval_samples_per_second": 76.159,
    "eval_steps_per_second": 12.693
    }


The scores are bad as we have trained on a tiny corpus. Training on a larger corpus should give good results.

