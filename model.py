"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-03-21

MODEL: BUILDING, TRAINING AND TESTING

Contains:
- build():  TensorFlow 2 model
- train():  Starts training. It's an Autograph function: the @tf.function decorator tranforms train_on_batch() inner
            function into its TensorFlow graph representation.
- test():   Checks model performance on Test data.
"""


def build(params):
    """
    Implements a seq2seq RNN with Convolutional self attention. It keeps a canonical
    Encoder-Decoder structure: an Embedding layers receives the sequence of chars and
    learns a representation. This series is received by two different layers at the same time.
    First, an LSTM Encoder layer, whose output is repeated and sent to the Decoder. Second, a
    block of 1D Conv layers. Their kernel filters work as multi-head self attention layers.
    All their scores are pushed through a TanH gate that scales each score in the [-1,1] range.
    Both LSTM and Conv outputs are concatenated and sent to an LSTM Decoder, that processes
    the signal and sents it to Dense layers, performing the prediction for each step of the
    output series.

    Args: params dict
    """
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import (
        Input, Embedding, LSTM, RepeatVector, Conv1D, BatchNormalization,
        Concatenate, LSTM, TimeDistributed, Dense
    )

    # ENCODER
    encoder_input = Input(shape = (params['len_input'],))

    encoder_embedding = Embedding(input_dim = params['dict_size'],
                                  output_dim = params['embedding_size'])(encoder_input)

    encoder_lstm = LSTM(params['len_input'], name = 'encoder_lstm')(encoder_embedding)

    encoder_output = RepeatVector(params['len_input'], name = 'encoder_output')(encoder_lstm)

    # Convolutional Self-Attention
    conv_1 = Conv1D(filters = params['conv_filters'][0], kernel_size = params['kernel_size'],
                    activation = params['conv_activation'], padding = 'same', name = 'conv1')(encoder_embedding)
    if params['use_batchnorm']:
        conv_1 = BatchNormalization(name = 'batchnorm_1')(conv_1)

    conv_2 = Conv1D(filters = params['conv_filters'][1], kernel_size = params['kernel_size'],
                    activation = params['conv_activation'], padding = 'same', name = 'conv2')(conv_1)
    if params['use_batchnorm']:
        conv_2 = BatchNormalization(name = 'batcnorm_2')(conv_2)

    conv_3 = Conv1D(filters = params['conv_filters'][2], kernel_size = params['kernel_size'],
                    activation = params['conv_activation'], padding = 'same', name = 'conv3')(conv_2)
    if params['use_batchnorm']:
        conv_3 = BatchNormalization(name = 'batchnorm_3')(conv_3)

    attention = tf.nn.tanh(conv_3, name = 'attention')

    # DECODER
    concatenation = Concatenate(axis = -1, name = 'concatenation')([encoder_output, attention])

    decoder_lstm = LSTM(params['len_input'], return_sequences = True,
                        name = 'decoder_lstm')(concatenation)

    decoder_dense = TimeDistributed(Dense(params['decoder_dense_units'],
                                          activation = params['decoder_dense_activation']),
                                    name = 'decoder_dense')(decoder_lstm)

    decoder_output = TimeDistributed(Dense(params['dict_size'], activation = params['decoder_output_activation']),
                                     name = 'decoder_output')(decoder_dense)


    model = Model(inputs = [encoder_input], outputs = [decoder_output])
    return model


def start_training(model, params, X_train, Y_train, X_val, Y_val):
    import time
    import numpy as np
    import tensorflow as tf
        
    optimizer = tf.keras.optimizers.Adam(learning_rate = params['learning_rate'])
    
    @tf.function
    def train_on_batch(model, X_batch, Y_batch):
        with tf.GradientTape() as tape:
            current_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    Y_batch, model(X_batch), from_logits = True))
        gradients = tape.gradient(current_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return current_loss
    
    
    for epoch in range(params['n_epochs']):
        start = time.time()

        for iteration in range(X_train.shape[0] // params['batch_size']):
            take = iteration * params['batch_size']
            X_batch = X_train[ take:take+params['batch_size'] , : ]
            Y_batch = Y_train[ take:take+params['batch_size'] , : ]
            
            current_loss = train_on_batch(model, X_batch, Y_batch)
            
            validation_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    Y_val, model(X_val), from_logits = True))

        print('{}.   \tTraining Loss: {}   \tValidation Loss: {}   \tTime: {}ss'.format(
            epoch,
            current_loss.numpy(),
            validation_loss.numpy(),
            round(time.time()-start, 2)
        ))
    print('Training complete.\n')
    model.save('{}/{}.h5'.format(params['save_path'], params['model_name']))

    print('Model saved at:\n\t{}'.format(params['save_path']))
    return None


def check_performance_on_test_set(model_path, X_test, Y_test):
    import numpy as np
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)

    #P_test = model.predict(X_test)

    test_loss = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(Y_test, model(X_test),
                                                        from_logits = True))

    print("\nTest Loss: {}".format(test_loss.numpy()))
    return None
