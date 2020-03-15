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
    from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
    
    return model




# ENCODER
encoder_input = Input()
encoder_embedding = Embedding(params['vocab_size'], params['embedding_size'])(encoder_input)
encoder_output, encoder_state = LSTM(params['encoder_units'], 
                                     return_sequences = True, 
                                     return_state = True, 
                                     recurrent_initializer = params['recurrent_initializer'])(encoder_embedding)


# ATTENTION
query_with_time_axis = tf.expand_dims(encoder_state, 1)

W1 = Dense(params['decoder_units'])(query_with_time_axis)
W2 = Dense(params['decoder_units'])(encoder_output)
score = tf.nn.tanh(W1 + W2)
score = Dense(1)(score)

attention_weights = tf.nn.softmax(score, axis = 1)

context_vector = attention_weights * values
context_vector = tf.reduce_sum(context_vector, axis = 1)


# DECODER

# CREDO che la x qui sia encoder_output, ma devo assicurarmene
decoder_input = Input()

x = Embedding(params['vocab_size'], params['embedding_size'])(x)
x = tf.concat([tf.expand_dims(context_vector, 1), x], axis = -1)

decoder_output, decoder_state = LSTM(params['decoder_units'], 
                                     return_sequences = True, 
                                     return_state = True, 
                                     recurrent_initializer = params['recurrent_initializer'])(x)

decoder_output = tf.reshape(decoder_output, (-1, decoder_output.shape[2]))

decoder_output = Dense(params['vocab_size'])(decoder_output)







# decoder(dec_input, dec_hidden, enc_output)




@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    
    with tf.GradientTape() as tape:
        
        enc_output, enc_hidden = encoder(inp)
        
        dec_hidden = enc_hidden
        
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * params['batch_size'], 1)
        
        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss


