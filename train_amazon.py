"""
Author: Ivan Bongiorni      2020-10-27

MODEL TRAINING
"""
from pdb import set_trace as BP


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

    # Local imports
    import model
    import tools.tools_amazon as tools

    # Load config params
    params = yaml.load(open(os.getcwd() + '/config.yaml'), yaml.Loader)

    # Set TensorFlow GPU configurations
    import tensorflow as tf
    tools.set_gpu_configurations(params)

    # Load char2idx dict to get 'vocab_size' parameter
    char2idx = yaml.load(open(os.getcwd() + '/data_processed/char2idx_amazon.yaml'), yaml.Loader)
    params['vocab_size'] = len(char2idx)+1  # additional +1 for 0 (right padding)

    # If model already exists, load it. Else make one
    if params['model_name']+'.h5' in os.listdir(os.getcwd()+'/saved_models/'):
        print('Loading model: "{}"'.format(params['model_name']))
        chatbot = tf.keras.models.load_model(os.getcwd()+'/saved_models/'+params['model_name']+'.h5')
    else:
        print('Creation of new model: "{}"'.format(params['model_name']))
        chatbot = model.build(params)
    chatbot.summary()

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
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])

    @tf.function
    def train_on_batch(Q_batch, A_batch):

        with tf.GradientTape() as tape:
            batch_loss = 0

            for i in range(A_batch.shape[0]):
                next_char_prediction = chatbot([Q_batch, A_batch[:,0:i+1]])  # Teacher forcing

                # compute loss of this specific char and add it to existing batch_loss
                batch_loss += loss(A_batch[:,i:i+1], next_char_prediction)

            batch_loss /= A_batch.shape[1]  # Mean Loss

        gradients = tape.gradient(batch_loss, chatbot.trainable_variables)
        optimizer.apply_gradients(zip(gradients, chatbot.trainable_variables))
        return batch_loss


    print('\nStart Training:')
    print('{} epochs x {} iterations.\n'.format(params['n_epochs'], len(filenames_train) // params['batch_size']))
    for epoch in range(params['n_epochs']):

        # Shuffle data by shuffling filenames array
        if params['shuffle']:
            filenames_train = filenames_train[ np.random.choice(filenames_train.shape[0], filenames_train.shape[0], replace=False) ]

        for iteration in range(filenames_train.shape[0] // params['batch_size']):
            # Fetch batch by filenames index and train
            start = iteration * params['batch_size']
            batch = [ np.load('{}/data_processed/Training/{}'.format(os.getcwd(), filename), allow_pickle=True) for filename in filenames_train[start:start+params['batch_size']] ]
            Q_batch = np.stack([ array[0] for array in batch ])
            A_batch = np.stack([ array[1] for array in batch ])

            # Train step
            training_loss = train_on_batch(Q_batch, A_batch)

            if iteration % 50 == 0:
                batch = np.random.choice(filenames_val, size=params['val_batch_size'], replace=False)
                batch = [ np.load('{}/data_processed/Validation/{}'.format(os.getcwd(), filename), allow_pickle=True) for filename in batch ]

                Q_batch = np.stack([ array[0] for array in batch ])
                A_batch = np.stack([ array[1] for array in batch ])

                validation_loss = 0
                for i in range(A_batch.shape[0]):
                    next_char_prediction = chatbot([Q_batch, A_batch[:,0:i+1]])
                    validation_loss += loss(A_batch[:,i:i+1], next_char_prediction)
                validation_loss /= A_batch.shape[1]

                print(f'{epoch}.{iteration}  \tTraining Loss: {training_loss}  \tValidation Loss: {validation_loss}')

    print('\nTraining complete.\n')

    chatbot.save(os.getcwd() + '/saved_models/' + params['model_name'] + '.h5')
    print('Model saved at:\n' + os.getcwd() + '/saved_models/' + params['model_name'] + '.h5')
    return None


if __name__ == '__main__':
    main()
