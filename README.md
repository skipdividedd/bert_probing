# Repository for probing research

## Our pipeline

1. Get training data.
2. ['Spoiled'](https://github.com/skipdividedd/bert_probing/blob/main/illiterate_text_deeppavlov.ipynb) data, so that adjective's gender is broken.
3. [Trained](https://github.com/skipdividedd/bert_probing/blob/main/rubert_training.ipynb) two Russian BERT models.
4. Marked data from [rusenteval](https://github.com/RussianNLP/RuSentEval) with stanza either per sent or per word.
5. Conducted 3 types of probing experiments (by CLS token, by mean sentence embedding and per token), largely relying on [NeuroX](https://github.com/fdalvi/NeuroX).
6. Compared the [results](https://github.com/skipdividedd/bert_probing/tree/main/html_visualisations).

## Probably useful

There are src files for probing and an example notebook. 
