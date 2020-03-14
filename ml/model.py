"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-03-14

MODEL BUILDING



build_model() is a wrapper, instantiating and assembling all the parts.
"""


params['vocab_size']
params['encoder_units']
params['decoder_units']



def build(params):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Embedding, LSTM, Dense
    
    return model



# ENCODER
x = Embedding(vocab_size, embedding_dim)(x)

output, state = LSTM(params['encoder_units'], 
                     return_sequences = True, 
                     return_state = True, 
                     recurrent_initializer='glorot_uniform')(x)


# ATTENTION
query_with_time_axis = tf.expand_dims(query, 1)

W1 = Dense(params['decoder_units'])(query_with_time_axis)
W2 = Dense(params['decoder_units'])(values) 
score = tf.nn.tanh(W1 + W2)
score = Dense(1)(score)

attention_weights = tf.nn.softmax(score, axis = 1)

context_vector = attention_weights * values
context_vector = tf.reduce_sum(context_vector, axis = 1)



# DECODER

context_vector, attention_weights = self.attention(hidden, enc_output)

x = Embedding(vocab_size, params['embedding_size'])(x)
x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

output, state = LSTM(params['decoder_units'], 
                     return_sequences = True, 
                     return_state = True, 
                     recurrent_initializer='glorot_uniform')(x)

output = tf.reshape(output, (-1, output.shape[2]))

x = Dense(vocab_size)(output)














