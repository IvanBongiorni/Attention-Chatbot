# WARNING: WORK IN PROGRESS

# Chatbot for Twitter Customer Support.

This is a Chatbot for Twitter Customer Support.

Data have been taken from Kaggle's [Customer Support on Twitter](https://www.kaggle.com/thoughtvector/customer-support-on-twitter) dataset. This dataset comprehends tweet exchanges from multiple companies. Each company requires a different model implementation; this Repository contains a separate section for each.


## How it works

The model is a **Seq2Seq** Neural Network with **LSTM** layers and **Bahdanau Attention** in **TensorFlow 2.1**.

( DETAILS )


## Structure of the Repository

/dataprep/
  `tools_\[company\].py`: Input pipe for \[company\]'s data. It generates ready-to-go Q and A matrices, together with a char2idx dictionary to maps chars and relative numerical indices.


## Modules

tensorflow = 2.0.1
numpy
pandas

