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

    # Import params and update path
    print('Importing configuration parameters.')
    params = yaml.load(open(os.getcwd() + '/config.yaml'), yaml.Loader)

    # Load dataset
    df = pd.read_csv(os.getcwd() + '/data_raw/twcs.csv')

    print('Creation of vectorized alphabet.')
    alphabet = tools_amazon.generate_alphabet()
    char2idx = { char[1]: char[0] for char in enumerate(alphabet, 1) }
    # Add start and end tokens
    char2idx['<END>'] = len(char2idx) + 1
    char2idx['<UNK>'] = len(char2idx) + 1
    char2idx['<START>'] = 0
    # Save dictionary for training later
    yaml.dump(char2idx, open(os.getcwd()+'/data_processed/char2idx_amazon.yaml', 'w'), default_flow_style=False)

    ## CLEANING
    # Pick clients' tweets that started a chat
    print('Cleaning text data.')
    exchanges = df[ df['inbound'] & (df['in_response_to_tweet_id'].isna()) ]

    # Attach relative answers
    exchanges = exchanges.merge(df[['author_id', 'text', 'in_response_to_tweet_id']],
                                left_on='tweet_id',
                                right_on='in_response_to_tweet_id')
    # Filter for current company
    exchanges = exchanges[ exchanges['author_id_y']=='AmazonHelp' ]
    exchanges = exchanges[['text_x', 'text_y']]  # Keep useful cols

    # Turn all lowercase
    exchanges['text_x'] = exchanges['text_x'].str.lower()
    exchanges['text_y'] = exchanges['text_y'].str.lower()

    # Keep only 'en' tweets
    exchanges['english_x'] = [ tools_amazon.check_language(tweet) for tweet in exchanges['text_x'].tolist() ]
    exchanges['english_y'] = [ tools_amazon.check_language(tweet) for tweet in exchanges['text_y'].tolist() ]
    exchanges = exchanges[ (exchanges['english_x'] == 'en') & (exchanges['english_y'] == 'en') ]
    exchanges.drop(['english_x', 'english_y'], axis = 1, inplace = True)

    # Keep just complete tweets - not (1/2) or (2/2)'s
    exchanges = exchanges[ (~exchanges['text_y'].str.endswith('(1/2)')) & (~exchanges['text_y'].str.endswith('(2/2)')) ]

    # Main text cleaning
    exchanges['text_x'] = [ tools_amazon.clean_text(tweet) for tweet in exchanges['text_x'].tolist() ]
    exchanges['text_y'] = [ tools_amazon.clean_text(tweet) for tweet in exchanges['text_y'].tolist() ]
    # Remove unknown signs from output
    exchanges['text_y'] = [ tools_amazon.process_y_text(tweet, alphabet) for tweet in exchanges['text_y'].tolist() ]

    ## VECTORIZE DATASET
    print('Character vectorization.')
    Q = [ tools_amazon.vectorize_tweet(tweet, char2idx) for tweet in exchanges['text_x'].tolist() ]
    A = [ tools_amazon.vectorize_tweet(tweet, char2idx) for tweet in exchanges['text_y'].tolist() ]

    ## PACK INTO FINAL ARRAYS
    # NaN-padding on the right based on max_length
    max_length = max(len(max(Q, key=len)), len(max(A, key=len)))
    Q = [ np.concatenate([ q, np.empty((max_length-len(q))) ]) for q in Q ]
    A = [ np.concatenate([ a, np.empty((max_length-len(a))) ]) for a in A ]

    Q = np.stack(Q)
    A = np.stack(A)
    Q = Q.astype(np.float32)
    A = A.astype(np.float32)

    ## TRAIN-VALIDATION-TEST SPLIT
    sample = np.random.choice(
        range(3),
        Q.shape[0],
        p = [1-np.sum(params['val_test_ratio']), params['val_test_ratio'][0], params['val_test_ratio'][1]],
        replace = True)
    Q_train = Q[ sample==0 , : ]
    A_train = A[ sample==0 , : ]
    Q_val = Q[ sample==1 , : ]
    A_val = A[ sample==1 , : ]
    Q_test = Q[ sample==2 , : ]
    A_test = A[ sample==2 , : ]

    ## TODO: Move to training part
    # # Expand A dimensions for compatibility with model output
    # A_train = np.expand_dims(A_train, axis=-1)
    # A_val = np.expand_dims(A_val, axis=-1)
    # A_test = np.expand_dims(A_test, axis=-1)

    ## SAVE
    Q_train = pd.DataFrame(Q_train)
    A_train = pd.DataFrame(A_train)
    Q_val = pd.DataFrame(Q_val)
    A_val = pd.DataFrame(A_val)
    Q_test = pd.DataFrame(Q_test)
    A_test = pd.DataFrame(A_test)

    Q_train.to_csv(os.getcwd() + '/data_processed/Q_train.csv', index = False, header = False)
    A_train.to_csv(os.getcwd() + '/data_processed/A_train.csv', index = False, header = False)
    Q_val.to_csv(os.getcwd() + '/data_processed/Q_val.csv', index = False, header = False)
    A_val.to_csv(os.getcwd() + '/data_processed/A_val.csv', index = False, header = False)
    Q_test.to_csv(os.getcwd() + '/data_processed/Q_test.csv', index = False, header = False)
    A_test.to_csv(os.getcwd() + '/data_processed/A_test.csv', index = False, header = False)

    print('\nData saved in /data_processed/* directories:')
    print('Train Q/A size:     ', Q_train.shape)
    print('Validation Q/A size:', Q_val.shape)
    print('Test Q/A size:      ', Q_test.shape)

    print('\nProcessing executed in {}ss.'.format(round(time.time()-start, 2)))
    return None


if __name__ == '__main__':
    main()
