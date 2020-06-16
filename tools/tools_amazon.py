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
