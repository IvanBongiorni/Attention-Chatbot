"""
Author: Ivan Bongiorni
2020-06-16
Repository:

MODEL TRAINING
"""


def main():
    import os
    import yaml
    import time
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    
    # local imports
    import model
    import tools.tools_amazon as tools
    
    tools.set_gpu_configurations()

    print('Loading configuration parameters.')
    params = yaml.load( open(os.getcwd() + '/config.yaml'), yaml.Loader )

    print('Loading preprocessed data.')
    X_train = pd.read_csv(os.getcwd()+'/data_processed/Q_train.csv')
    Y_train = pd.read_csv(os.getcwd()+'/data_processed/A_train.csv')
    X_val = pd.read_csv(os.getcwd()+'/data_processed/Q_val.csv')
    Y_val = pd.read_csv(os.getcwd()+'/data_processed/A_val.csv')

    # If model already exists, load it. Else make one
    if params['model_name']+'.h5' in os.listdir(os.getcwd()+'/saved_models/'):
        print('Loading model: "{}"'.format(params['model_name']))
        chatbot = tf.keras.models.load_model(os.getcwd()+'/saved_models/'+params['model_name']+'.h5')
    else:
        print('Creation of new model: "{}"'.format(params['model_name']))
        chatbot = model.build(params)

    optimizer = tf.keras.optimizers.Adam(learning_rate = params['learning_rate'])

    @tf.function
    def train_on_batch(X_batch, Y_batch):
        with tf.GradientTape() as tape:
            current_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    Y_batch, chatbot(X_batch), from_logits = True))
        gradients = tape.gradient(current_loss, chatbot.trainable_variables)
        optimizer.apply_gradients(zip(gradients, chatbot.trainable_variables))
        return current_loss

    print('\nStart Training:')
    print('{} epochs x {} iterations.\n'.format(params['n_epochs'], X_train.shape[0] // params['batch_size']))

    for epoch in range(params['n_epochs']):
        start = time.time()

        if params['shuffle']:
            index = np.random.choice(X_train.shape[0], size = X_train.shape[0], replace = False)

        for iteration in range(X_train.shape[0] // params['batch_size']):
            take = index[iteration * params['batch_size']]
            X_batch = X_train[take]
            Y_batch = Y_train[take]

            training_loss = train_on_batch(X_batch, Y_batch)

        # To spare memory, compute val_loss on random subset of Val data
        val_sample = np.random.choice(X_val.shape[0], size = params['val_batch_size'], replace = False)
        validation_loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                Y_val[ val_sample , : ], chatbot(X_val[ val_sample , : ]), from_logits = True))

        print('{}.   \tTraining Loss: {}   \tValidation Loss: {}   \tTime: {}ss'.format(
            epoch,
            training_loss.numpy(),
            validation_loss.numpy(),
            round(time.time()-start, 2)
        ))
    print('\nTraining complete.\n')

    chatbot.save(os.getcwd()+'/saved_models/'+params['model_name']+'.h5')
    print('Model saved at:\n' + os.getcwd()+'/saved_models/'+params['model_name']+'.h5')
    return None


if __name__ == '__main__':
    main()
