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
    
#     print('controllo params')
#     print(params['data_path'])
#     print(params['save_path'])
#     print(params['load_saved_model'])
    
    # Load data
    print('Loading and preprocessing data:')
    Q_train, A_train, Q_val, A_val, Q_test, A_test, char2idx = tools_amazon.get_amazon_dataset(params)
    params['dict_size'] = len(char2idx)  # add as hyperparams to build model
    
    # Instantiate model
    if params['load_saved_model']:
        print('Loading model from:\n{}'.format(current_path + '/saved_models/'))
        model = tf.keras.models.load_model(current_path + '/saved_models/' + params['model_name'])
    else:
        print('Building model.')
        model = model.build(params)
        print('Model implemented as:\n')
        model.summary()
    
    # Train model
    print('\nStart model training for {} epochs'.format(params['n_epochs']))
    model.train(model, params, Q_train, A_train, Q_val, A_val)
    
    # Test model
    print("\n\nTesting model's performance on unseen data:")
    model.test(current_path + '/saved_models/' + params['model_name'], 
               Q_test, A_test)
    return None


if __name__ == '__main__':
    start()