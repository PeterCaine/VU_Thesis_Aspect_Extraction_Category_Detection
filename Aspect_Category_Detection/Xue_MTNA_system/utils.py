import pickle
import re
import numpy as np
from numpy import array
from numpy import asarray
from numpy import zeros

import keras
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Bidirectional, Input, Embedding, LSTM,\
     Dense, merge, Conv1D, GlobalMaxPooling1D, Concatenate, concatenate
from keras.models import Model
from sklearn.metrics import classification_report

def pickle_loader (path):
    """loads stanford CoreNLP lemmatised texts from pickle

    Args:
        path (path ): pickles are pre-processed CoreNLP files

    Returns:
        .pkl: [lemmatised pickle files loades as binary data]
    """
    return pickle.load(open(path, 'rb'))

def integer_encoder (df_train, df_gold):
    """integer encodes lemmatised texts for input in embeddings layer

    Args:
        df_train (dataframe ): SemEval data as dataframe
        df_gold (dataframe): SemEval datas as dataframe

    Returns:
        varied: padded docs and padded_test_docs are integer encoded review texts, t is the tokenizer and vocab size 
        is the limit of the vocabulary size for use in NN
    """

    # define documents - these are lemmatised texts in df_train and df_gold
    docs = df_train.lemmatised.tolist()
    test_docs = df_gold.lemmatised.tolist()
    # find max sentence length for max_lenght
    # len(max(docs, key = lambda i: len(i)))
    # define class labels
    labels = df_train.category0.tolist()
    # prepare tokenizer
    t = Tokenizer()
    # tokenize texts
    t.fit_on_texts(docs)
    vocab_size = len(t.word_index) + 1
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(docs)
    encoded_test_docs = t.texts_to_sequences(test_docs)
    # print(encoded_docs)
    # pad documents to a max length of 80 words
    max_length = 80
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post', truncating = 'post')
    padded_test_docs = pad_sequences(encoded_test_docs, maxlen=max_length, padding='post', truncating = 'post')
    return padded_docs, padded_test_docs, t, vocab_size

def glove_loader (path_to_glove):
    """loads GloVe 200d vectors into memory and creates lookup index

    Args:
        path_to_glove (string): system path to GloVe embeddings

    Returns:
        index: a lookup table for words to corresponding embeddings values
    """ 
    embeddings_index = dict()
    f = open (path_to_glove, encoding = 'utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print(f'Loaded {len(embeddings_index)} word vectors.')
    return embeddings_index

def substitue_embeddings(embeddings_index, t, vocab_size):
    """takes lookup index, tokenizer and limit of vocab to create embedding matrix for entry into 
    NN

    Args:
        embeddings_index (index matrix): lookup index for words and corresponding embeddings values
        t (tokenizer): keras pre-processing tokenizer
        vocab_size (int): max size of value 

    Returns:
        matrix: vocabulary vectorized for entry into NN
    """
    # create a weight matrix for words in training docs
    embedding_matrix = zeros((vocab_size, 200))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector  
    return embedding_matrix
    
def populate_df_with_binary_cols_per_label (df_train, df_test):
    """takes dataframe and adds 1 or 0 per row in label column depending on presence or not of a category 
    in the review

    Args:
        df_test (dataframe): SemEval test data
        df_train (dataframe): SemEval training data

    Returns:
        dataframe: updated with columns for category labels (either 1 or 0)
    """
    df_test = df_test[['text', 'category0', 'category1','category2','category3']]
    df_test['cat_ambience'] = 0
    df_test['cat_anecdotes/miscellaneous'] = 0
    df_test['cat_food'] = 0
    df_test['cat_price'] = 0
    df_test['cat_service'] = 0
    df_test.loc[(df_test['category0'] == 'ambience') | (df_test['category1'] == 'ambience') | (df_test['category2'] == 'ambience')| (df_test['category3'] == 'ambience'),'cat_ambience'] = 1
    df_test.loc[(df_test['category0'] == 'anecdotes/miscellaneous') | (df_test['category1'] == 'anecdotes/miscellaneous') | (df_test['category2'] == 'anecdotes/miscellaneous')| (df_test['category3'] == 'anecdotes/miscellaneous'),'cat_anecdotes/miscellaneous'] = 1
    df_test.loc[(df_test['category0'] == 'food') | (df_test['category1'] == 'food') | (df_test['category2'] == 'food')| (df_test['category3'] == 'food'),'cat_food'] = 1
    df_test.loc[(df_test['category0'] == 'price') | (df_test['category1'] == 'price') | (df_test['category2'] == 'price')| (df_test['category3'] == 'price'),'cat_price'] = 1
    df_test.loc[(df_test['category0'] == 'service') | (df_test['category1'] == 'service') | (df_test['category2'] == 'service')| (df_test['category3'] == 'service'),'cat_service'] = 1
    df_train = df_train[['text', 'category0', 'category1','category2','category3']]
    df_train['cat_ambience'] = 0
    df_train['cat_anecdotes/miscellaneous'] = 0
    df_train['cat_food'] = 0
    df_train['cat_price'] = 0
    df_train['cat_service'] = 0
    df_train.loc[(df_train['category0'] == 'ambience') | (df_train['category1'] == 'ambience') | (df_train['category2'] == 'ambience')| (df_train['category3'] == 'ambience'),'cat_ambience'] = 1
    df_train.loc[(df_train['category0'] == 'anecdotes/miscellaneous') | (df_train['category1'] == 'anecdotes/miscellaneous') | (df_train['category2'] == 'anecdotes/miscellaneous')| (df_train['category3'] == 'anecdotes/miscellaneous'),'cat_anecdotes/miscellaneous'] = 1
    df_train.loc[(df_train['category0'] == 'food') | (df_train['category1'] == 'food') | (df_train['category2'] == 'food')| (df_train['category3'] == 'food'),'cat_food'] = 1
    df_train.loc[(df_train['category0'] == 'price') | (df_train['category1'] == 'price') | (df_train['category2'] == 'price')| (df_train['category3'] == 'price'),'cat_price'] = 1
    df_train.loc[(df_train['category0'] == 'service') | (df_train['category1'] == 'service') | (df_train['category2'] == 'service')| (df_train['category3'] == 'service'),'cat_service'] = 1
    
    return df_train, df_test

def run_mtna(embedding_matrix, vocab_size):
    """implementation of Xue et al.'s MTNA-s NN system

    Args:
        embedding_matrix (matrix): values for words in texts and embeddings encoded for entry into NN
        vocab_size (int): max size of vocab

    Returns:
        NN model: trained model 
    """
    main_input = Input(shape=(80,), name='main_input')

    x = Embedding(output_dim=200, input_dim=vocab_size, input_length=80,weights=[embedding_matrix],trainable=True)(main_input)
    lstm_out = Bidirectional(LSTM(units = 300, activation = 'tanh', dropout = 0.2, recurrent_dropout = 0.2, return_sequences = True))(x)

    cnn5_in = Conv1D(filters = 100, kernel_size = 5, padding='same', activation='tanh',strides=1,input_shape = (None, 300))(lstm_out)
    cnn5_max_out = GlobalMaxPooling1D()(cnn5_in)
    bi_max_out = GlobalMaxPooling1D()(lstm_out)
    merge_one = concatenate([bi_max_out, cnn5_max_out])
    final_out = Dense(2, activation='softmax')(merge_one)
    model = keras.Model(inputs=main_input, outputs=final_out)

    model.compile(optimizer='Adadelta', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())
    return model

def predict (model, padded_docs, padded_test_docs, df_train, df_test):
    """runs 5 1 vs 1 binary classifiers using MTNA-s NN model. Prints classification report for each label. 

    Args:
        model (NN model): MTNA-s
        padded_docs (vector): vector representations of texts
        padded_test_docs (vector): vector representations of test data
        df_train (dataframe): review texts and corresponding labels in df
        df_test (datafraeme): review texts and corresponding labels in df
    """
    model.fit(padded_docs, df_train['cat_ambience'].tolist(), epochs=20, validation_split=0.2, verbose = False)
    output_list = model.predict(padded_test_docs)
    system_predictions_amb = pd.DataFrame(np.argmax(output_list,axis=1), columns=['Predicted'])
    print('ambience', '\n',classification_report(df_test['cat_ambience'].tolist(), system_predictions_amb))
    
    model.fit(padded_docs, df_train['cat_anecdotes/miscellaneous'].tolist(), epochs=20, validation_split=0.2, verbose = False)
    output_list = model.predict(padded_test_docs)
    system_predictions_anec = pd.DataFrame(np.argmax(output_list,axis=1), columns=['Predicted'])
    print('anecdotes/miscellaneous', '\n',classification_report(df_test['cat_anecdotes/miscellaneous'].tolist(), system_predictions_anec))
    
    model.fit(padded_docs, df_train['cat_food'].tolist(), epochs=20, validation_split=0.2, verbose = False)
    output_list = model.predict(padded_test_docs)
    system_predictions_food = pd.DataFrame(np.argmax(output_list,axis=1), columns=['Predicted'])
    print('food', '\n',classification_report(df_test['cat_food'].tolist(), system_predictions_food))
    
    model.fit(padded_docs, df_train['cat_price'].tolist(), epochs=20, validation_split=0.2, verbose = False)
    output_list = model.predict(padded_test_docs)
    system_predictions_price = pd.DataFrame(np.argmax(output_list,axis=1), columns=['Predicted'])
    print('price', '\n',classification_report(df_test['cat_price'].tolist(), system_predictions_price))
    
    model.fit(padded_docs, df_train['cat_service'].tolist(), epochs=20, validation_split=0.2, verbose = False)
    output_list = model.predict(padded_test_docs)
    system_predictions_service = pd.DataFrame(np.argmax(output_list,axis=1), columns=['Predicted'])
    print('service', '\n',classification_report(df_test['cat_service'].tolist(), system_predictions_service))
