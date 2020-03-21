"""
This script runs a trained Chatbot through terminal interface.
"""
import tensorflow as tf













if __name__ == '__main__':
    path_to_trained_model = None
    if path_to_trained_model:
        talk(path_to_trained_model)
    else:
        ## TODO: go get params dict
        talk(params['save_model_path'])
