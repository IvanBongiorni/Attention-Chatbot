"""
Author: Ivan Bongiorni,     https://github.com/IvanBongiorni
2020-06-15

DATA PRE PROCESSING PIPELINE

This is meant to be run before training pieline. It loads a dataframe stored at
/data_raw/twcs.csv and processes and saves Train, Validation and Test data in
/data_processed/* directories.
"""
from pdb import set_trace as BP


def main(verbose = True):
    import os
    import yaml
    import time
    import numpy as np
    import pandas as pd

    # Local imports
    from tools import tools_amazon
    import model

    print('\nStart data pre-processing for @AmazonHelp.\n')
    start = time.time()

    ## LOAD data and config
    print('Importing dataset and configuration parameters.')
    df = pd.read_csv(os.getcwd() + '/data_raw/twcs.csv')
    params = yaml.load(open(os.getcwd() + '/config.yaml'), yaml.Loader)

    ## ALPHABET creation, vectorization and saving
    print('Creation of vectorized alphabet.')
    alphabet = tools_amazon.generate_alphabet()

    # Add START, END tokens, and UNK for unknown tokens-chars
    alphabet += ['<START>']
    alphabet += ['<UNK>']
    alphabet += ['<END>']

    # Start enumerating and 1, use 0 for padding
    char2idx = { char[1]: char[0] for char in enumerate(alphabet, 1) }

    # Save dictionary for training later
    yaml.dump(char2idx, open(os.getcwd()+'/data_processed/char2idx_amazon.yaml', 'w'), default_flow_style=False)

    ## DATA CLEANING
    # Pick clients' tweets that started a chat
    print('Cleaning text data.')
    exchanges = df[ df['inbound'] & (df['in_response_to_tweet_id'].isna()) ]

    # Attach relative answers
    exchanges = exchanges.merge(df[['author_id', 'text', 'in_response_to_tweet_id']], left_on='tweet_id', right_on='in_response_to_tweet_id')

    # Filter for current company
    exchanges = exchanges[ exchanges['author_id_y']=='AmazonHelp' ]
    exchanges = exchanges[['text_x', 'text_y']]  # Keep only useful cols

    # Turn all lowercase
    exchanges['text_x'] = exchanges['text_x'].str.lower()
    exchanges['text_y'] = exchanges['text_y'].str.lower()

    # Keep only 'en' tweets
    exchanges['english_x'] = [ tools_amazon.check_language(tweet) for tweet in exchanges['text_x'].tolist() ]
    exchanges['english_y'] = [ tools_amazon.check_language(tweet) for tweet in exchanges['text_y'].tolist() ]
    exchanges = exchanges[ (exchanges['english_x']=='en') & (exchanges['english_y']=='en') ]
    exchanges.drop(['english_x', 'english_y'], axis=1, inplace=True)

    # Keep just complete tweets - not (1/2) or (2/2)'s
    exchanges = exchanges[ (~exchanges['text_y'].str.endswith('(1/2)')) & (~exchanges['text_y'].str.endswith('(2/2)')) ]

    # Main text cleaning (regex based)
    exchanges['text_x'] = [ tools_amazon.clean_text(tweet) for tweet in exchanges['text_x'].tolist() ]
    exchanges['text_y'] = [ tools_amazon.clean_text(tweet) for tweet in exchanges['text_y'].tolist() ]
    # Remove unknown signs from output
    exchanges['text_y'] = [ tools_amazon.process_y_text(tweet, alphabet) for tweet in exchanges['text_y'].tolist() ]

    ## VECTORIZE DATASET and RIGHT-ZERO PAD
    print('Character vectorization.')
    Q = [ tools_amazon.vectorize_tweet(tweet, char2idx) for tweet in exchanges['text_x'].tolist() ]
    A = [ tools_amazon.vectorize_tweet(tweet, char2idx) for tweet in exchanges['text_y'].tolist() ]

    max_length = max(len(max(Q, key=len)), len(max(A, key=len)))
    Q = [ np.concatenate([ q, np.zeros((max_length-len(q))) ]) for q in Q ]
    A = [ np.concatenate([ a, np.zeros((max_length-len(a))) ]) for a in A ]

    ## SAVE in single (q,a) couples, to facilitace batching during training
    # First, pack Q and A whole matrices to facilitate Train-Val_test splits
    Q = np.stack(Q)
    A = np.stack(A)
    Q = Q.astype(np.float32)
    A = A.astype(np.float32)

    sample = np.random.choice(
        range(3),
        len(Q),
        p = [1-np.sum(params['val_test_ratio']), params['val_test_ratio'][0], params['val_test_ratio'][1]],
        replace = True)
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
