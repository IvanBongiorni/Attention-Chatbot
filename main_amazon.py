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
    # Disable TensorFlow warnings
    import logging
    tf.get_logger().setLevel(logging.ERROR)

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
    params['dict_size'] = len(char2idx)  # add as hyperparams to build model


    print('\n\n')
    print('Q_train:', Q_train.shape)
    print('A_train:', A_train.shape)

    print('Q_val:', Q_val.shape)
    print('A_val:', A_val.shape)

    print('Q_test:', Q_test.shape)
    print('A_test:', A_test.shape)
    print('\n\n')

    # Instantiate model
    if params['load_saved_model']:
        print('Loading model from:\n{}'.format(current_path + '/saved_models/'))
        seq2seq = tf.keras.models.load_model(current_path + '/saved_models/' + params['model_name'])
    else:
        print('Building model.')
        seq2seq = model.build(params)
        print('Model implemented as:\n')
        seq2seq.summary()


    print('\n\nControlla la dimensione del dizionario e i valori in Q e A')
    BP()


    # Train model
    print('\nStart model training for {} epochs'.format(params['n_epochs']))
    model.train(seq2seq, params, Q_train, A_train, Q_val, A_val)

    # Test model
    print("\n\nTesting model's performance on unseen data:")
    model.test(current_path + '/saved_models/' + params['model_name'],
               Q_test, A_test)
    return None


if __name__ == '__main__':
    start()
