"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-03-21

MODEL: BUILDING, TRAINING AND TESTING

Contains:
- build():  TensorFlow 2 model
- train():  Starts training. It's an Autograph function: the @tf.function decorator
            tranforms train_on_batch() inner function into its TensorFlow graph representation.
- test():   Checks model performance on Test data.
"""
import time
import numpy as np
import tensorflow as tf

# Local imports
import tools.tools_amazon as tools


def build(params):
    '''
    Seq2seq RNN with Luong's multiplicative attention mechanism.
    '''
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Embedding, LSTM, Attention, Concatenate, Dense

    # Encoder receives tokenized input, generates representation and gives it to LSTM
    encoder_input = Input(shape=(None,))
    encoder_embedding = Embedding(input_dim=params['vocab_size'], output_dim=params['embedding_size'])(encoder_input)
    encoder_lstm = LSTM(params['encoder_lstm_units'])(encoder_embedding)

    # Decoder receives answer through teacher forcing, then generates representation
    # the LSTM layer produces a representation for the next char
    decoder_input = Input(shape=(None,))
    decoder_embedding = Embedding(input_dim=params['vocab_size'], output_dim=params['embedding_size'])(decoder_input)
    decoder_lstm = LSTM(params['decoder_lstm_units'])(decoder_embedding)

    # Multiplicative Decoder pays attention to Encoder to produce next char prediction
    attention = Attention()([decoder_lstm, encoder_lstm])
    context = Concatenate()([decoder_lstm, attention])
    decoder_output = Dense(params['vocab_size'], activation='softmax')(context)

    model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_output, attention])
    return model


### TODO: MUST BE REWRITTEN COMPLETELY

# def check_performance_on_test_set(model_path, X_test, Y_test):
#     import numpy as np
#     import tensorflow as tf
#
#     model = tf.keras.models.load_model(model_path)
#
#     # P_test = model.predict(X_test)
#     test_loss = tf.reduce_mean(
#         tf.keras.losses.sparse_categorical_crossentropy(Y_test, model(X_test),
#                                                         from_logits = True))
#
#     print("\nTest Loss: {}".format(test_loss.numpy()))
#     return None
