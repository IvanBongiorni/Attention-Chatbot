"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-03-21

KEEPS TRAINING PIPELINE TOGETHER

It imports params and raw data, runs processing pipeline, builds/loads and trains
model, saves it in specific folder.
To be run from shell.
"""

def start(verbose = True):
    import os
    import yaml
    import time
    from pdb import set_trace as BP
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import tensorflow as tf
    # Solves Convolution CuDNN error
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    # Disable TensorFlow warnings
    # import logging
    # tf.get_logger().setLevel(logging.ERROR)

    # Local modules
    from dataprep import tools_amazon
    import model

    # Get current path
    current_path = os.getcwd()

    # Import params and update path
    print('\n\nImporting configuration parameters.')
    params = yaml.load(open(current_path + '/config.yaml'), yaml.Loader)

    params['data_path'] = current_path + '/data/'
    params['save_path'] = current_path + '/saved_models/'

    # Load data
    print('Loading and preprocessing data:')
    Q_train, A_train, Q_val, A_val, Q_test, A_test, char2idx = tools_amazon.get_amazon_dataset(params)

    # ########################
    # Q_train = pd.DataFrame(Q_train)
    # Q_train.to_pickle('/home/ivan/Documents/ML/projects/nlp/chatbot/temp_data/Q_train.pkl')
    # A_train = pd.DataFrame(np.squeeze(A_train))
    # A_train.to_pickle('/home/ivan/Documents/ML/projects/nlp/chatbot/temp_data/A_train.pkl')
    # Q_val = pd.DataFrame(Q_val)
    # Q_val.to_pickle('/home/ivan/Documents/ML/projects/nlp/chatbot/temp_data/Q_val.pkl')
    # A_val = pd.DataFrame(np.squeeze(A_val))
    # A_val.to_pickle('/home/ivan/Documents/ML/projects/nlp/chatbot/temp_data/A_val.pkl')
    # ########################
    #
    # print('\n\nINTERRUZIONE PER SALVATAGGIO:')
    # BP()

    params['dict_size'] = len(char2idx)  # add as hyperparams to build model

    # Instantiate model
    if params['load_saved_model']:
        print('Loading model from:\n{}'.format(current_path + '/saved_models/'))
        seq2seq = tf.keras.models.load_model(current_path + '/saved_models/' + params['model_name'])
    else:
        print('Building model.')
        seq2seq = model.build(params)
        print('Model implemented as:\n')
        seq2seq.summary()

    # Train model
    print('\nStart model training for {} epochs'.format(params['n_epochs']))
    model.start_training(seq2seq, params, Q_train, A_train, Q_val, A_val)

    # Test model
    print("\n\nTesting model's performance on unseen data:")
    model.check_performance_on_test_set(current_path + '/saved_models/' + params['model_name'] + '.h5',
                                        Q_test, A_test)
    return None


if __name__ == '__main__':
    start()
