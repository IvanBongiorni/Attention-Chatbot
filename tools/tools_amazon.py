"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-03-08

DATA PRE-PROCESSING TOOLBOX FOR amazonhelp TWEETS

The code is intended to be used as follows: it's a local module containing a main wrapper function: get_Amazon_dataset().
It should be called from the main Notebook or script, provided the right path to the Twitter Customer Support Dataset
in order to get the already vectorized Q and A matrices. Additionally, a Python dictionary char2idx is returned too,
providing a mapping between characters and numerical indexes.

Data are pre-processed for character embedding RNNs.
"""

def set_gpu_configurations(params):
    ''' Avoid repetition of this GPU setting block '''
    import tensorflow as tf
    
    print('Setting GPU configurations.')
    # This block avoids GPU configuration errors
    if params['use_gpu']:
        # This prevents CuDNN 'Failed to get convolution algorithm' error
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        # To see list of allocated tensors in case of OOM
        tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)
    else:
        try:
            # Disable all GPUs
            tf.config.set_visible_devices([], 'GPU')
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != 'GPU'
        except:
            print('Invalid device or cannot modify virtual devices once initialized.')
        pass
    return None


def generate_alphabet():
    import string

    alphabet = string.printable
    alphabet = alphabet.replace('ABCDEFGHIJKLMNOPQRSTUVWXYZ', '')
    alphabet = alphabet.replace('[\\]^_`{|}~', '')
    alphabet = alphabet.replace('\t\n\r\x0b\x0c', '')
    alphabet = alphabet.replace(';<=>', '')
    alphabet = alphabet.replace('*+', '')
    # alphabet += '§' # ö'

    alphabet = list(alphabet)

    return alphabet


def check_language(tweet):
    ''' Uses langdetect to check for English tweets. Non-English tweets won't be
    considered. List remove_for_langdetect is meant to fix recurrent errors in
    Japanese tweets that get mistaken for English. '''
    import langdetect

    remove_for_langdetect = [
        'amazon fire tv stick', 'fire tv stick', 'amazonfiretvstick', 'amazonmusicunlimited',
        'amazon kindle unlimited', 'amazon echo dot', 'prime music'
    ]

    # this block prevents some 'ja' to be detected as 'en'
    if any([ element in tweet for element in remove_for_langdetect ]):
        for element in remove_for_langdetect:
            tweet = tweet.replace(element, '')

    # return 'en' for no language tweets like single emoji's
    try:
        return langdetect.detect(tweet)
    except:
        return 'en'


def clean_text(tweet):
    '''
    This is the main, initial pre-processing function of texts. It executes the
    following steps:
        - Removes unwanted chars, such as utf-8 and emoji's
        - Replaces URLs with char §
        - Replaces order numbers with char ö
        - Collapses multiple spaces into one, and trims final str
    (It is to be iterated on whole dataset column with list comprehension.)

    Further processing must be differentiated between Customers and Company tweets.
    '''
    import string
    import re
    import numpy as np
    import pandas as pd

    # Removal of unwanted chars / patterns
    tweet = tweet.replace('\t', ' ') # space or new line chars to subst with space
    tweet = tweet.replace('\n', ' ')
    tweet = tweet.replace('\r', ' ')
    tweet = tweet.replace('\x0b', ' ')
    tweet = tweet.replace('\x0c', ' ')
    tweet = tweet.replace('\u200b', ' ')
    tweet = tweet.replace('\u200d', ' ')
    tweet = tweet.replace(';', ",")
    tweet = tweet.replace('‘', "'")
    tweet = tweet.replace('‘', "'")
    tweet = tweet.replace('´', "'")
    tweet = tweet.replace('`', "'")
    tweet = tweet.replace('’', "'")
    tweet = tweet.replace('”', "'")
    tweet = tweet.replace('“', "'")

    tweet = tweet.replace('\{', "\(")
    tweet = tweet.replace('\}', "\)")
    tweet = tweet.replace('\[', "\(")
    tweet = tweet.replace('\]', "\)")

    tweet = tweet.replace('&amp;', '&')
    tweet = tweet.replace('@amazonhelp', '') # Remove '@amazonhelp'

    tweet = re.sub(r'@[0-9]+', '', tweet) # reference id's (e.g. '@12345')
    tweet = re.sub(r'(\^\w*)$', '', tweet) # Remove final signature (e.g.: ... ^ib)
    tweet = tweet.replace('\u200d♂️', '') # recurrent two emoji codes
    tweet = tweet.replace('\u200d♀️', '')

    # Removes ref to tweet parts, e.g.: "... (1/2)"
    tweet = re.sub(r"\([0-9]/[0-9]\)", "", tweet)

    # Substitution of elements with anonymized tags
    tweet = re.sub(r'https?://\S+|www\.\S+', '§', tweet) # Replace URLs with char §
    # tweet = re.sub(r'\w{3}-\w{7}-\w{7}', 'ö', tweet) # Replace order numbers with char ö

    tweet = re.sub(' +', ' ', tweet)  # collapse multiple spaces left into one
    tweet = tweet.strip() # trim left and right spaces
    return tweet


def process_y_text(tweet, alphabet):
    ''' Processing of Response: all chars that are not in the chosen
    alphabet will be eliminated. Eventual multiple spaces will be collapsed '''
    import re

    tweet = [ char if char in alphabet else ' ' for char in tweet ]
    tweet = ''.join(tweet)

    tweet = re.sub(' +', ' ', tweet)  # collapse multiple spaces left into one
    tweet = tweet.strip() # trim left and right spaces
    return tweet


def vectorize_tweet(tweet, char2idx):
    ''' chars --> np.array(idx)    To be iterated by vectorize_dataset().
    Out-of-alphabet chars set to 0 for RNN masking. '''
    import numpy as np

    chars_vector = list(tweet)
    chars_vector = [ char2idx[char] if char in char2idx.keys() else char2idx['<UNK>'] for char in chars_vector ]
    chars_vector = [char2idx['<START>']] + chars_vector + [char2idx['<END>']] # add start and end tokens
    chars_vector = np.array(chars_vector)
    return chars_vector
