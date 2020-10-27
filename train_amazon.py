"""
Author: Ivan Bongiorni      2020-10-27

MODEL TRAINING
"""


def main():
    '''
    Model training. Loads configs and lists of Train and Val filenames to sample
    mini batches.
    Each model is either loaded if found in /saved_models, or implemented from
    scratch.
    At each iteration a (Q,A) mini batch is built and trained with custom train
    function (Autograph). Print models performance on Validation periodically.
    Model is then saved (overwritten) in /saved_models.
    '''
    import os
    import yaml
    import time
    import numpy as np

    # Set TensorFlow GPU configurations
    import tensorflow as tf
    tools.set_gpu_configurations()

    # Local imports
    import model
    import tools.tools_amazon as tools

    # Load config params
    params = yaml.load(open(os.getcwd() + '/config.yaml'), yaml.Loader)

    # If model already exists, load it. Else make one
    if params['model_name']+'.h5' in os.listdir(os.getcwd()+'/saved_models/'):
        print('Loading model: "{}"'.format(params['model_name']))
        chatbot = tf.keras.models.load_model(os.getcwd()+'/saved_models/'+params['model_name']+'.h5')
    else:
        print('Creation of new model: "{}"'.format(params['model_name']))
        chatbot = model.build(params)

    # List all Train an Validation files
    filenames_train = os.listdir(os.getcwd() + '/data_processed/Training/')
    if 'readme_training.md' in filenames_train: filenames_train.remove('readme_training.md')
    if '.gitignore' in filenames_train: filenames_train.remove('.gitignore')
    filenames_train = np.array(filenames_train)

    filenames_val = os.listdir(os.getcwd() + '/data_processed/Validation/')
    if 'readme_validation.md' in filenames_val: filenames_val.remove('readme_validation.md')
    if '.gitignore' in filenames_val: filenames_val.remove('.gitignore')
    filenames_val = np.array(filenames_val)

    # Define optimizer and train step function - with Loss
    optimizer = tf.keras.optimizers.Adam(learning_rate = params['learning_rate'])

    @tf.function
    def train_on_batch(Q_batch, A_batch):
        with tf.GradientTape() as tape:
            current_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(A_batch, chatbot(Q_batch), from_logits=True))
        gradients = tape.gradient(current_loss, chatbot.trainable_variables)
        optimizer.apply_gradients(zip(gradients, chatbot.trainable_variables))
        return current_loss

    print('\nStart Training:')
    print('{} epochs x {} iterations.\n'.format(params['n_epochs'], X_train.shape[0] // params['batch_size']))

    for epoch in range(params['n_epochs']):

        # Shuffle data by shuffling filenames array
        if params['shuffle']:
            filenames_train = filenames_train[ np.random.choice(filenames_train.shape[0], filenames_train.shape[0], replace=False) ]

        for iteration in range(filenames_train.shape[0] // params['batch_size']):
            # Fetch batch by filenames index and train
            start = iteration * params['batch_size']
            batch = [ np.load('{}/data_processed/Training/{}'.format(os.getcwd(), filename), allow_pickle=True) for filename in X_files[start:start+params['batch_size']] ]
            Q_batch = np.concatenate([ array[0] for array in batch ])
            A_batch = np.concatenate([ array[1] for array in batch ])

            # Train step
            training_loss = train_on_batch(Q_batch, A_batch)

        if iteration % 50 == 0:
                batch = np.random.choice(filenames_val, size=params['val_batch_size'], replace=False)
                batch = [ np.load('{}/data_processed/Validation/{}'.format(os.getcwd(), filename), allow_pickle=True) for filename in filenames_val[start:start+params['val_batch_size']] ]

                Q_batch = np.concatenate([ array[0] for array in batch ])
                A_batch = np.concatenate([ array[1] for array in batch ])

                validation_loss = tf.reduce_mean(
                    tf.keras.losses.sparse_categorical_crossentropy(A_batch, chatbot(Q_batch), from_logits=True))

                print('{}.{}   \tTraining Loss: {}   \tValidation Loss: {}'.format(
                    epoch, iteration, current_loss, validation_loss))

    print('\nTraining complete.\n')

    chatbot.save(os.getcwd() + '/saved_models/' + params['model_name'] + '.h5')
    print('Model saved at:\n' + os.getcwd() + '/saved_models/' + params['model_name'] + '.h5')
    return None


if __name__ == '__main__':
    main()
