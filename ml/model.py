"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-03-14

MODEL BUILDING

This is a seq2seq model with Convolutional pseudo-Attention. It differs from canonical Attention since 
its scores are not calculated iteratively, as the output of the model progresses. All scores are computed
at once, stacked with the Encoder's output, and fed into the Decoder LSTM layer.

With it I expect to be able to put together two benefits of RNNs and CNNs. RNNs are perfect in order to 
process time dependent data, but their problem is dealing with long-term dependency: all Encoder output 
information has to be squeezed into a single vector. 1D Conv layers are meant intestead to process time series 
in a different way, and to "bring time back" to the Decoder.
"""


def build(params):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Embedding, LSTM, RepeatVector, Conv1D, Dense, TimeDistributed
    

    # ENCODER
    encoder_input = Input()
    
    encoder_embedding = Embedding(params['vocab_size'], params['embedding_size'])(encoder_input)
    
    encoder_output = LSTM(params['encoder_lstm_units'], 
                          return_sequences = True, 
                          recurrent_initializer = params['recurrent_initializer'])(encoder_embedding)
    
    encoder_output = RepeatVector(params['decoder_units'])
    
    
    # Convolutional Pseudo-Attention
    conv_1 = Conv1D(filters = params['conv_filters'][0], 
                    kernel_size = 3, padding = 'same', activation = params['conv_activation'])
    conv_2 = Conv1D(filters = params['conv_filters'][1], 
                    kernel_size = 3, padding = 'same', activation = params['conv_activation'])
    conv_3 = Conv1D(filters = params['conv_filters'][0], 
                    kernel_size = 3, padding = 'same', activation = params['conv_activation'])
    # scores = Dense(params['decoder_units'], activation = 'softmax')(conv)
    
    
    # Concatenation of Encoder and Pseudo-Attention scores
    concatenation = Concatenate([encoder_output, scores], axis = )
    
    
    # DECODER
    decoder_output = LSTM(params['decoder_lstm_units'],
                          return_sequences = True,
                          recurrent_initializer = params['recurrent_initializer'])(concatenation)
    
    decoder_dense_1 = TimeDistributed(Dense(params['decoder_dense_units'], 
                                            activation = params['decoder_dense_activation']))(decoder_output)
    decoder_dense_2 = TimeDistributed(Dense(params['decoder_dense_units'], 
                                            activation = params['decoder_dense_activation']))(decoder_dense_1)
    decoder_output = TimeDistributed(Dense(1, activation = None))(decoder_dense_2)
    
    
    model = Model(inputs = [encoder_input], outputs = [decoder_output])
    
    return model




