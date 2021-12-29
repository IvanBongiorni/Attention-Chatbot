"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-10-24

DATA PRE-PROCESSING PIPELINE

This is meant to be run before training pieline. Loads raw dataframe stored as
/data_raw/twcs.csv and processes tweets by company. It saves individual (q, a)
processed tweet couples in /data_processed/Training, /Validation and /Test folders
as .npy files.
"""
import os
import yaml
import time
from pdb import set_trace as BP

import numpy as np
import pandas as pd
import langdetect

import nltk

# Local imports
from tools import tools_amazon
import model


def generate_vectorized_vocabulary():
    import os

    # go get raw data

    # create corpus and extract words


    # create vocabulary dict with standard tokens
    word2idx = {}

    word2idx['<START>']
    word2idx['<UNK>']
    word2idx['<END>']

    word2idx

    return word2idx


def check_language(tweet):
    ''' Uses langdetect to check for English tweets. Non-English tweets won't be
    considered. List remove_for_langdetect is meant to fix recurrent errors in
    Japanese tweets that get mistaken for English. '''
    import langdetect

    remove_for_langdetect = [
        'amazon fire tv stick', 'fire tv stick', 'amazonfiretvstick', 'amazonmusicunlimited',
        'amazon kindle unlimited', 'amazon echo dot', 'prime music'
    ]

    if any([ element in tweet for element in remove_for_langdetect ]):
        for element in remove_for_langdetect:
            tweet = tweet.replace(element, '')

    # return 'en' for no language tweets (like emojis only)
    try:
        return langdetect.detect(tweet)
    except:
        return 'en'


def clean_tweet(tweet):
    import re
    from nltk import word_tokenize

    tweet = tweet.lower()  # lowercase

    # Remove URLs
    tweet = re.sub(r'http?://\S+|www\.\S+', 'URL', tweet)
    tweet = re.sub(r'https?://\S+|www\.\S+', 'URL', tweet)
    tweet = re.sub(r'@[a-zA-z0-9]+', 'USER', tweet)

    # Distance punctuation to facilitate tokenization
    tweet = tweet.replace(r"&amp;", " & ")
    tweet = tweet.replace(r"&quot;", ' " ')
    tweet = tweet.replace(r"&apos;", " ' ")
    tweet = tweet.replace(r"&gt;", " > ")
    tweet = tweet.replace(r"&lt;", " < ")
    tweet = tweet.replace(r'.', ' . ')
    tweet = tweet.replace(r';', ' ; ')
    tweet = tweet.replace(r':', ' : ')

    tweet = tweet.replace('\t', ' ')
    tweet = tweet.replace('\n', ' ')
    tweet = tweet.replace('\r', ' ')
    tweet = tweet.replace('\x0b', ' ')
    tweet = tweet.replace('\x0c', ' ')
    tweet = tweet.replace('\u200b', ' ')
    tweet = tweet.replace('\u200d', ' ')
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
    tweet = re.sub(r'(\^\w*)$', '', tweet) # Remove final signature (e.g.: ... ^ib)
    tweet = re.sub(r"\([0-9]/[0-9]\)", "", tweet)  # reference to tweet parts, e.g.: "... (1/2)"
    tweet = re.sub(r"[0-9]/[0-9]", "", tweet) # or "... 1/2"
    tweet = tweet.replace('\u200d♂️', '') # recurrent two emoji codes
    tweet = tweet.replace('\u200d♀️', '')

    tweet = re.sub(r' +', ' ', tweet)  # collapse multiple spaces into one
    tweet = tweet.replace(r'#', '')  # remove '#'

    tweet = nltk.word_tokenize(tweet)  # apply nltk tokenizer

    return tweet



def main():
    import os
    import yaml
    import time

    print('\nStart data pre-processing for @AmazonHelp.\n')
    start = time.time()

    print('Importing dataset and configuration parameters.')
    df = pd.read_csv(os.getcwd() + '/data_raw/twcs.csv')
    params = yaml.load(open(os.getcwd() + '/config.yaml'), yaml.Loader)

    print('Creation of vectorized vocabulary.')
    word2idx = generate__vectorized_vocabulary()

    # Save dictionary for training later
    yaml.dump(
        word2idx,
        open( os.path.join(os.getcwd(), 'data_processed', 'vocabulary_amazon.yaml'), 'w'),
        default_flow_style=False
    )

    print('Processing text data.')
    # Pick clients' tweets that started a chat
    exchanges = df[ df['inbound'] & (df['in_response_to_tweet_id'].isna()) ]

    # Attach relative answers
    qa = qa.merge(
        df[['author_id', 'text', 'in_response_to_tweet_id']],
        left_on='tweet_id',
        right_on='in_response_to_tweet_id'
    )

    # Filter for current company
    qa = qa[ qa['author_id_y']=='AmazonHelp' ]
    qa = qa[['text_x', 'text_y']]  # Keep only useful cols

    # Keep only 'en' tweets
    qa['english_x'] = [ check_language(tweet) for tweet in qa['text_x'] ]
    qa['english_y'] = [ check_language(tweet) for tweet in qa['text_y'] ]
    qa = qa[ (qa['english_x']=='en') & (qa['english_y']=='en') ]
    qa.drop(['english_x', 'english_y'], axis=1, inplace=True)

    # Keep just complete tweets - not (1/2) or (2/2)'s
    qa = qa[ (~qa['text_y'].str.endswith('(1/2)')) & (~qa['text_y'].str.endswith('(2/2)')) ]

    print('Cleaning text data and vectorization.')
    Q = [ clean_text(tweet) for tweet in qa['text_x'] ]
    A = [ clean_text(tweet) for tweet in qa['text_y'] ]

    # Right zero pad
    Q = [ np.concatenate([ q, np.zeros((params['seq_len']-len(q))) ]) for q in Q ]
    A = [ np.concatenate([ a, np.zeros((params['seq_len']-len(a))) ]) for a in A ]

    # Train - Val - Test splits
    Q = np.stack(Q)
    A = np.stack(A)
    Q = Q.astype(np.float32)
    A = A.astype(np.float32)

    sample = np.random.choice(
        range(3),
        len(Q),
        p = [ 1-np.sum(params['val_test_ratio']), params['val_test_ratio'][0], params['val_test_ratio'][1] ],
        replace = True
    )

    Q_train = Q[ sample==0 , : ]
    A_train = A[ sample==0 , : ]

    Q_val = Q[ sample==1 , : ]
    A_val = A[ sample==1 , : ]

    Q_test = Q[ sample==2 , : ]
    A_test = A[ sample==2 , : ]

    # Then pack each (q,a) couple and save them
    for i in range(Q_train.shape[0]):
        array = np.array([ Q_train[i,:], A_train[i,:] ])
        np.save(os.getcwd()+'/data_processed/Training/X_{}'.format(str(i).zfill(6)), array)

    for i in range(Q_val.shape[0]):
        array = np.array([ Q_val[i,:], A_val[i,:] ])
        np.save(os.getcwd()+'/data_processed/Validation/X_{}'.format(str(i).zfill(6)), array)

    for i in range(Q_test.shape[0]):
        array = np.array([ Q_test[i,:], A_test[i,:] ])
        np.save(os.getcwd()+'/data_processed/Test/X_{}'.format(str(i).zfill(6)), array)

    print('\nProcessing executed in {}ss.'.format(round(time.time()-start, 2)))

    return None


if __name__ == '__main__':
    main()
