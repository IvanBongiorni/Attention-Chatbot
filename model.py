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
    Seq2seq RNN with Multiplicative attention mechanism.

    The model must be thought as a couple of ANNs: Encoder and Decoder. Since its a
    seq2seq task, each has its Input() layer - for Q and A in this specific case.
    After Encoder produces an output, Decoder receives its hidden states and produces
    its own. The two outputs are combined, multiplicatively, to get attention scores
    and a context vector. It gets concatenated to Decoder's LSTM output, and fed to
    a final Dense layer with softmax output for next char prediction.

    NB: Event though an Attention() layer is already available in tf.keras (implementing
    Luong's multiplicative attention) I won't use it for RNNs. As reported in the official
    docs, that level is suitable for dense and conv nets only, and not for RNNs.
    That is why I have implemented my own Attention mechanism.
    '''
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Embedding, LSTM, Dot, Activation, Attention, Concatenate, Dense

    # Encoder receives tokenized input, generates representation and gives it to LSTM
    # Internal staset are returned, to be fed into Decoder LSTM.
    encoder_input = Input(shape=(None,))
    encoder_embedding = Embedding(input_dim=params['vocab_size'], output_dim=params['embedding_size'], name='Encoder Embedding')(encoder_input)
    encoder_lstm_output, encoder_h, encoder_c = LSTM(64, return_state=True, name='Encoder_LSTM')(encoder_embedding)

    # Decoder receives answer through teacher forcing, then generates representation
    # the LSTM layer produces a representation for the next char
    decoder_input = Input(shape=(None,))
    decoder_embedding = Embedding(input_dim=params['vocab_size'], output_dim=params['embedding_size'], name='Decoder Embedding')(decoder_input)
    decoder_lstm_output = LSTM(64, name='Decoder_LSTM')(decoder_embedding, initial_state=[encoder_h, encoder_c])

    # Multiplicative Attention attends Encoder and Decoder LSTM outputs
    attention = encoder_lstm_output * decoder_lstm_output
    attention = Activation('softmax', name='Attention')(attention)

    # The context vector is then combined with original Decoder LSTM output for prediction
    context_vector = attention * encoder_lstm_output
    decoder_combined_context = Concatenate()([decoder_lstm_output, context_vector])

    decoder_dense_output = Dense(65, activation='softmax', name='Output layer')(decoder_combined_context)

    # model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_output, attention])
    model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_dense_output)
    return model


# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Embedding, LSTM, Attention, Concatenate, Dense
#
# # Encoder receives tokenized input, generates representation and gives it to LSTM
# encoder_input = Input(shape=(None,))
# encoder_embedding = Embedding(input_dim=params['vocab_size'], output_dim=params['embedding_size'])(encoder_input)
# encoder_output, encoder_h, encoder_c = LSTM(params['encoder_lstm_units'], return_states=True)(encoder_embedding)
#
# # Decoder receives answer through teacher forcing, then generates representation
# # the LSTM layer produces a representation for the next char
# decoder_input = Input(shape=(None,))
# decoder_embedding = Embedding(input_dim=params['vocab_size'], output_dim=params['embedding_size'])(decoder_input)
# decoder_lstm = LSTM(params['decoder_lstm_units'])(decoder_embedding, initial_state=[encoder_h, encoder_c])
#
# # Multiplicative Decoder pays attention to Encoder to produce next char prediction
# attention = Attention()([decoder_lstm, encoder_lstm])
# context = Concatenate()([decoder_lstm, attention])
#
#
# attention = dot([decoder, encoder], axes=[2, 2])
# attention = Activation('softmax')(attention)
#
#
# decoder_output = Dense(params['vocab_size'], activation='softmax')(context)
#
# model = Model(inputs=[encoder_input, decoder_input], outputs=[decoder_output, attention])
# return model
