"""
This script runs a trained Chatbot through terminal interface.

It calls a talk() function that implements a Q/A exchange between user and @amazonhelp bot.
THe model called is the one specified in params['model_name']. talk() calles itself
recursively if the chat is not shut.

Functions process_question() and process_answer() are faster copies of preprocessing
functions from tools_amazon.py.
"""
from pdb import set_trace as BP

def process_question(tweet, char2idx, alphabet):
    ''' Repeats processing of Q tweets '''
    import numpy as np
    from tools.tools_amazon import clean_text, vectorize_tweet

    # tokenize chars
    tweet = clean_text(tweet)
    tweet = vectorize_tweet(tweet, char2idx)

    # right-zero pad and reshape to (1, 280)
    if len(tweet) < 280:
        tweet = np.concatenate([ tweet, np.zeros((280-len(tweet))) ])
    tweet = tweet[ np.newaxis , ... ]
    return tweet


def process_answer(answer, idx2char):
    ''' Repeats processing of A tweets '''
    import re
    import numpy

    answer = [ idx2char[char] for char in answer ]
    answer = ''.join(answer)
    answer = answer.replace('<START>', '')
    answer = answer.replace('<UNK>', '')
    answer = answer.replace('<END>', '')
    answer = answer.replace('§', '<LINK>')
    answer = re.sub(' +', ' ', answer)
    return answer


def talk():
    ''' Starts chat in Terminal '''
    import os
    # # Suppress tensorflow warnings
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import warnings
    warnings.simplefilter("ignore")
    import yaml
    import time
    import numpy as np
    import tensorflow as tf

    from tools.tools_amazon import generate_alphabet

    # Load config params
    params = yaml.load(open(os.getcwd() + '/config.yaml'), yaml.Loader)

    # BP()

    # Load char2idx dict to get 'vocab_size' parameter
    char2idx = yaml.load(open(os.getcwd() + '/data_processed/char2idx_amazon.yaml'), yaml.Loader)
    idx2char = {v: k for k, v in char2idx.items()}  # invert

    alphabet = generate_alphabet()

    print(f"""\nInteraction with '{params['model_name']}' for @amazonhelp.""")
    print("""[Limit your tweet to 280 chars. Type 'quit' to exit ]""")

    model = tf.keras.models.load_model(os.getcwd() + '/saved_models/{}.h5'.format(params['model_name']))

    chat_ongoing = True

    while chat_ongoing:
        tweet = input('@User:\t')

        # Check if tweet is too long or user didn't write anything and repeat loop
        if len(tweet) > 280:
            print('\n[ Tweet is larger than 280 chars, please type again. ]')
            continue
        elif len(tweet) == 0:
            print('\n[ Please type a message ]')
            continue

        # If tweet is OK then
        tweet = tweet.strip()  # basic cleaning
        if tweet.lower() == 'quit':
            print('\nShutting Chatbot down.')
            print('Conversation closed.')
            chat_ongoing = False
        else:
            tweet = process_question(tweet, char2idx, alphabet)

            ## NB: Only one char is generated each time. Answering requires iteration of model predictions
            ## while loop keeps generating chars autoregressively until '<END>' token is reached or max len

            # answer array of token indexes
            answer = np.array([char2idx['<START>']]).reshape((1,1))
            a = char2idx['<START>']

            counter = 0

            # while a != char2idx['<END>'] or answer.shape[1] < 280:
            while answer.shape[1] < 280:
                a = model.predict([tweet, answer.reshape((1, answer.shape[1]))])
                a = np.argmax(a).reshape((1,1))

                counter += 1
                print('\n', counter, '\t', a, '\t\t', a.shape, answer.shape)

                answer = np.hstack([answer, a])

            # At the end convert it to final text
            answer = process_answer(list(answer[0,:]), idx2char)

            # delay answer slightly (to look more realistic)
            if params['chat_sleep_time']>0:
                time.sleep(np.random.uniform(low=0.5, high=params['chat_sleep_time']))
            print('@amazonhelp:\t{}'.format(answer))

    return None



if __name__ == '__main__':
    talk()
