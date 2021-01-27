# WARNING: WORK IN PROGRESS
This repository is not ready yet. Please don't clone or use right now. A functioning prototype should be ready in the next weeks.

# Attention-based Chatbot for Twitter Customer Support.

This is a Chatbot for Twitter Customer Support.
It is implemented as a **Seq2seq RNN with Multiplicative Attention** mechanism.

Data have been taken from Kaggle's [Customer Support on Twitter](https://www.kaggle.com/thoughtvector/customer-support-on-twitter) dataset. 
This dataset comprehends tweet exchanges from multiple companies. 
Each company requires a different model implementation.


## How it works

The model implemented is a **Seq2Seq** Neural Network with **LSTM** layers and **Luong's Multiplicative Attention**. 
It is written in **TensorFlow 2.1** and optimized with **Autograph**.
The model is character-based, i.e. single characters are tokenized and predicted.
Training is implemented with **teacher forcing**.


## Structure of the Repository

Folders:
- `/data_raw`: uncompressed raw dataset must be pasted here.
- `/data_processed`: all pre-processed observations will be saved in `/Training`, `/Validation` and `/Test` sub-folders. 
It contains also `.yaml` dictionaries to translate from token (character) to vector; their naming convention is `char2idx_{company}.yaml`.
- `/dataprep`: contains all data preprocessing scripts, with names as `dataprep_{company}.py`.
- `/tools`: useful functions to be iterated in dataprep. 
One main `tools.py` module contains functions used for all models. 
For more company-specific tools other modules are available as `tools_{company}.py`.
- `/saved_models`: where trained models are saved and/or loaded after launching `train.py`.
- `/talk`: contains a list of scripts to be called from terminal to chat with a trained model. Naming convention is still `talk_{company}.py`.

Files:
- `config.yaml`: main configuration file. Every hyperparameter and model choice can be decided here.
- `model.py`: model implementation.
- `train.py`: model training. The company's model and the data to be trained on can be chosed from `config.yaml`.


## Modules
```
langdetect==1.0.8
tensorflow==2.1.0
numpy==1.18.1
pandas==1.0.1
```

## Bibliography

Papers:
- *Luong, Minh-Thang, Hieu Pham, and Christopher D. Manning. "Effective approaches to attention-based neural machine translation." arXiv preprint arXiv:1508.04025 (2015).*

Other useful resources:
- The official TensorFlow tutorial on a [Neural machine translation with attention](https://www.tensorflow.org/tutorials/text/nmt_with_attention).
- [Attention-based Sequence-to-Sequence in Keras](https://wanasit.github.io/attention-based-sequence-to-sequence-in-keras.html) on Wanasit Tanakitrungruang's [blog](https://wanasit.github.io/).
