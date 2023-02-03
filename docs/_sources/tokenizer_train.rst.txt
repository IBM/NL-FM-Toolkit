Training a Tokenizer from Scratch
======================================

Let us now look into a short tutorial on training a tokenizer from scratch. All the programs are run from the root folder of the repository.

To train a tokenizer we need a corpus. For this tutorial, we provide a sample corpus in the following folder. 

.. code-block:: console
   :linenos:

    $ ls demo/data/lm/
    english_sample.txt

The sample snippet of the corpus is here

.. code-block:: console
   :linenos:

    $ head demo/data/lm/english_sample.txt
    The Project Gutenberg eBook of Romeo and Juliet, by William Shakespeare

    This eBook is for the use of anyone anywhere in the United States and
    most other parts of the world at no cost and with almost no restrictions
    whatsoever. You may copy it, give it away or re-use it under the terms
    of the Project Gutenberg License included with this eBook or online at
    www.gutenberg.org. If you are not located in the United States, you
    will have to check the laws of the country where you are located before
    using this eBook.

    $ wc demo/data/lm/english_sample.txt
    2136   10152   56796 demo/data/lm/english_sample.txt


This text is extracted from Romeo and Juliet play by William Shakespeare from the Gutenberg Corpus ( https://www.gutenberg.org/cache/epub/1513/pg1513.txt )


We will train a Wordpiece tokenizer with a vocab size of around ``500``. The smaller vocab size is due to the corpus being small.

.. code-block:: console
   :linenos:

   $ python src/tokenizer/train_tokenizer.py \
        --input_file demo/data/lm/english_sample.txt \
        --name demo/model/tokenizer/ \
        --tokenizer_type wordpiece \
        --vocab_size 500

    [00:00:00] Pre-processing files (0 Mo)              ██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████                100%
    [00:00:00] Tokenize words                           ██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 4252     /     4252
    [00:00:00] Count pairs                              ██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 4252     /     4252
    [00:00:00] Compute merges                           ██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 387      /      387


The following files will be created inside ``demo/model/tokenizer/`` folder

.. code-block:: console
   :linenos:

   $ ls demo/model/tokenizer/
   tokenizer.json
   

Creating Model Configuration File
======================================

By default the `train_tokenizer.py` script doesn't create the model configuration files. The configuration file is required to load the model from `AutoTokenizer.from_pretrained()`. We now use the script `create_config.py` script to create the configuration file.


.. code-block:: console
    :linenos:

    $ python create_config.py \
        --path demo/model/tokenizer/ \
        --type gpt2
