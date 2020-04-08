"""
This script runs a trained Chatbot through terminal interface.

It calls a talk() function that implements a Q/A exchange between user and @amazonhelp bot.
THe model called is the one specified in params['model_name']. talk() calles itself
recursively if the chat is not shut.

Functions process_question() and process_answer() are faster copies of preprocessing
functions from tools_amazon.py.
"""

### PREPROCESS input Tweet
def process_question(tweet):
    return tweet


### PROCESS output Tweet
def process_answer(tweet):
    return tweet


def talk(params):
    ''' Starts chat in Terminal '''
    import time
    import numpy as np
    import tensorflow as tf

    print('\nInteraction with {} for @amazonhelp.'.format(params['model_name']))
    print("\t[Type 'Quit' to exit chat]")

    ### TODO: Load model

    chat_ongoing = True

    while chat_ongoing:
        tweet = input('@User:\t')
        tweet = tweet.strip()  # basic cleaning

        if tweet == 'Quit':
            print('\nShutting Chatbot down.')
            print('Conversation closed.')
            chat_ongoing = False
        elif not tweet:
            print('[ Please type a message ]')
        else:
            tweet = process_question(tweet)
            answer = model.predict(tweet)
            answer = process_answer(tweet)

            time.sleep(np.random.randint(0.3, 1))  # delay answer slightly
            print('@amazonhelp:\t{}'.format(answer))

    return None



if __name__ == '__main__':

    # Load config params
    import os
    import yaml

    talk(params)
