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
    ### BLOCCO TEMPORANEO PER VELOCIZZARE PIPELINE
    # import pickle
    # pickle.dump(Q_train, open("/home/ivan/Documents/ML/projects/NLP/chatbot/Q_train.pkl", "wb"))
    # pickle.dump(A_train, open("/home/ivan/Documents/ML/projects/NLP/chatbot/A_train.pkl", "wb"))
    # pickle.dump(Q_val, open("/home/ivan/Documents/ML/projects/NLP/chatbot/Q_val.pkl", "wb"))
    # pickle.dump(A_test, open("/home/ivan/Documents/ML/projects/NLP/chatbot/A_val.pkl", "wb"))
    # pickle.dump(Q_test, open("/home/ivan/Documents/ML/projects/NLP/chatbot/Q_test.pkl", "wb"))
    # pickle.dump(A_test, open("/home/ivan/Documents/ML/projects/NLP/chatbot/A_test.pkl", "wb"))
    # pickle.dump(char2idx, open("/home/ivan/Documents/ML/projects/NLP/chatbot/char2idx.pkl", "wb"))
    # Q_train = pickle.load(open("/home/ivan/Documents/ML/projects/NLP/chatbot/Q_train.pkl"))
    # A_train = pickle.dump(open("/home/ivan/Documents/ML/projects/NLP/chatbot/A_train.pkl"))
    # Q_val = pickle.dump(open("/home/ivan/Documents/ML/projects/NLP/chatbot/Q_val.pkl"))
    # A_test = pickle.dump(open("/home/ivan/Documents/ML/projects/NLP/chatbot/A_val.pkl"))
    # Q_test = pickle.dump(open("/home/ivan/Documents/ML/projects/NLP/chatbot/Q_test.pkl"))
    # A_test = pickle.dump(open("/home/ivan/Documents/ML/projects/NLP/chatbot/A_test.pkl"))
    # char2idx = pickle.dump(open("/home/ivan/Documents/ML/projects/NLP/chatbot/char2idx.pkl"))

    params['dict_size'] = len(char2idx)  # add as hyperparams to build model

    # print('\n\nDOPO PICKLE LOAD:\n')
    # BP()


    # print('\n\n')
    # print('Q_train:', Q_train.shape)
    # print('A_train:', A_train.shape)
    #
    # print('Q_val:', Q_val.shape)
    # print('A_val:', A_val.shape)
    #
    # print('Q_test:', Q_test.shape)
    # print('A_test:', A_test.shape)
    # print('\n\n')

    # Instantiate model
    if params['load_saved_model']:
        print('Loading model from:\n{}'.format(current_path + '/saved_models/'))
        seq2seq = tf.keras.models.load_model(current_path + '/saved_models/' + params['model_name'])
    else:
        print('Building model.')
        seq2seq = model.build(params)
        print('Model implemented as:\n')
        seq2seq.summary()


    # print('\n\nControlla la dimensione del dizionario e i valori in Q e A')
    # print('\n\nControlla i valori interni di Q e A')
    # print('\n\nControlla params[len_input]')
    # BP()


    # Train model
    print('\nStart model training for {} epochs'.format(params['n_epochs']))
    model.start_training(seq2seq, params, Q_train, A_train, Q_val, A_val)

    # Test model
    print("\n\nTesting model's performance on unseen data:")
    model.check_performance_on_test_set(current_path + '/saved_models/' + params['model_name'],
                                        Q_test, A_test)
    return None


if __name__ == '__main__':
    start()
