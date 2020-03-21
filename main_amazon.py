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
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    
    # Local modules
    from dataprep import tools_amazon
    from ml import model, train
    
    # Get current path
    current_path = os.getcwd()
    
    # Import params and update path
    params = yaml.load(current_path + '/config.yaml')
    params['data_path'] = current_path + '/data/'
    params['save_path'] = current_path + '/saved_models/'
    
    # Load data
    Q_train, A_train, Q_val, A_val, Q_test, A_test, char2idx = tools_amazon.get_amazon_dataset(path = params['data_path'])
    params['dict_size'] = len(char2idx)  # add as hyperparams to build model
    
    # Instantiate model
    if params['load_saved_model']:
        model = tf.keras.models.load_model(current_path + '/saved_models/' + params['model_name'])
    else:
        model = model.build(params)
        print('Model implemented:\n')
        model.summary()
    
    # Train model
    print('\nStart model training for {} epochs'.format(params['n_epochs']))
    train.start(model, params, Q_train, A_train, Q_val, A_val)
    
    return None




if __name__ == '__main__':
    start()