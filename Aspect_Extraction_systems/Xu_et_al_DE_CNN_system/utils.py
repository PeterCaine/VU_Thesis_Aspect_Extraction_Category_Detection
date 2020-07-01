import numpy as np
import pandas as pd
import gensim
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, GlobalMaxPooling1D 
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer

def load_test_train (path_train, path_test):
    
    # load in CoNLL formatted restaurant tsv 
    df_train = pd.read_csv(path_train, sep = '\t')
    df_test = pd.read_csv(path_test, sep = '\t')
    
    return df_train, df_test

def embed_swap (word_embedding_model, dataframe_series):
    '''takes a word embedding model and a dataframe series or words
    returns a list of embeddings in the same order as the series.
    '''
    embeddings = []
    for token in dataframe_series:
        if token in word_embedding_model:
            vector = word_embedding_model[token]
        else:
            vector = [0]*300
        embeddings.append(vector)
    return embeddings

def load_embeddings (w2v_path, xuv_path):
    
    w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    xuv = gensim.models.fasttext.load_facebook_model(xuv_path) 
    return w2v, xuv
    

def labels_to_categorical (dataframe_series):
    '''changes a series of labels into numerical categories and a dictionary to look up the cats
    
    '''
    labels = dataframe_series.tolist()
    label_set = set()
    for label in labels:
        label_set.add(label)
    label2Idx = {}
    for label in label_set:
        label2Idx[label] = len(label2Idx)
    map_prep = pd.DataFrame(labels)
    mapped = list(map_prep[0].map(label2Idx))
    Y_label = np.asarray(mapped)

    label2Idx
    return Y_label, label2Idx

def training_data (general_embed_model, custom_embed_model, dataframe_series, label_series):
    """[summary]

    Args:
        general_embed_model ([type]): [description]
        custom_embed_model ([type]): [description]
        dataframe_series ([type]): [description]
        label_series ([type]): [description]

    Returns:
        [type]: [description]
    """
    embeddings = embed_swap (general_embed_model, dataframe_series)
    x_train = np.array(embeddings)
    x_train= np.reshape(x_train, (1, x_train.shape[0], x_train.shape[1]))
    train_labels, label_index = labels_to_categorical(label_series)
    y_train = keras.utils.np_utils.to_categorical(train_labels)
    y_train = np.reshape(y_train,(1,y_train.shape[0], y_train.shape[1]))
    domain_embed = embed_swap (custom_embed_model, dataframe_series.fillna('na'))
    x_domain_train = np.array(domain_embed)
    x_domain_train= np.reshape(x_domain_train, (1, x_domain_train.shape[0], x_domain_train.shape[1]))
    X_train = np.concatenate((x_domain_train, x_train), axis=2)
    x_valid = X_train[:,31684:,:]
    X = X_train[:,:31684,:]
    # print('shape: xvalid :', x_valid.shape)
    # print('shape: X :', X.shape)
    y_valid = y_train[:,31684:,:]
    y = y_train[:,:31684,:]
    # print('shape: y_valid :', y_valid.shape)
    # print('shape: y :', y.shape)
    return X, x_valid, y_train, y_valid

def test_data (general_embed_model, custom_embed_model, gold_series, gold_label_series):
    """[summary]

    Args:
        general_embed_model ([type]): [description]
        custom_embed_model ([type]): [description]
        gold_series ([type]): [description]
        gold_label_series ([type]): [description]

    Returns:
        [type]: [description]
    """
    test_embeddings = embed_swap (general_embed_model, gold_series)
    test_labels, label_index = labels_to_categorical(gold_label_series)
    x_test = np.array(test_embeddings)
    x_test= np.reshape(x_test, (1, x_test.shape[0], x_test.shape[1]))
    domain_test = embed_swap (custom_embed_model, gold_series.fillna('na'))
    x_domain_test = np.array(domain_test)
    x_domain_test= np.reshape(x_domain_test, (1, x_domain_test.shape[0], x_domain_test.shape[1]))
    X_test = np.concatenate((x_domain_test, x_test), axis=2)
    # print ('X_test.shape: ', X_test.shape)
    return X_test,label_index

def run_cnn (X_train, y_train, X_valid, y_valid, embedding_dims, batch_size,
             kernel_size, epochs, verbose = False):
    """[summary]

    Args:
        X_train ([type]): [description]
        y_train ([type]): [description]
        X_valid ([type]): [description]
        y_valid ([type]): [description]
        embedding_dims ([type]): [description]
        batch_size ([type]): [description]
        kernel_size ([type]): [description]
        epochs ([type]): [description]
        verbose (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """
    model = Sequential()
    model.add(Conv1D(filters = 128, kernel_size = kernel_size,padding='same', activation='relu',strides=1,input_shape = (None, embedding_dims)))
    model.add(Dropout(0.55,))
    model.add(Conv1D(filters = 128, kernel_size = kernel_size, padding='same', activation='relu'))
    model.add(Conv1D(filters = 256, kernel_size = kernel_size, padding='same'))
    model.add(Conv1D(filters = 256, kernel_size = kernel_size, padding='same'))
    # model.add(Conv1D(filters = 256, kernel_size = kernel_size, padding='same'))
    model.add(Dense(units=3, activation='softmax'))
    if verbose == True:
        model.summary()
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data = (X_valid, y_valid), verbose = verbose)
    return model

def predict (model, df_test, label_index):
    """[summary]

    Args:
        model ([type]): [description]
        df_test ([type]): [description]
        label_index ([type]): [description]
    """
    
    output_list = model.predict_classes(X_test)
    gold_series = df_test.label.tolist()
    system_predictions = output_list[0]
    d = {'gold':gold_series,'predicted':system_predictions}
    result = pd.DataFrame(d)
    inv_map = {v: k for k, v in label_index.items()}
    predictions = result.predicted.map(inv_map)
    df_test['predicted'] = predictions
    target_names = ['B','I','O']
    print(classification_report(result.gold, df_test.predicted, target_names=target_names))
    df_test['gold_bi']= result.gold.replace({'I': 'B'})
    df_test['predicted_bi']= df_test.predicted.replace({'I': 'B'})
    target_names = ['B','O']
    print(classification_report(df_test.gold_bi, df_test.predicted_bi, target_names=target_names))