import xml.etree.ElementTree as et
import numpy as np
import re

import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report



def xml_category_df (xml_file):
    """takes SemEval 2014 dataset, extracts aspect terms and aspect categories per
    review and reformats them as a dataframe with each term and category having a 
    separate column 

    Args:
        xml_file ([file.xml]): SemEval 2014 gold or training xml (restaurants domain
        used here)

    Returns:
        [dataframe]: pd dataframe with each row a separate review and each col 
        containing either a single aspect term (phrase) or aspect category associated
        with the review
    """
    xtree = et.parse(xml_file)
    xroot = xtree.getroot()
    outer_dict = {}
    for n, node in enumerate(xroot):
        inner_dict = {}
        inner_dict['text']= node[0].text
        try:
            aspect_category_list = []
            for i, term in enumerate(node[1].iter('aspectTerm')):
                inner_dict['aspect'+str(i)]=term.attrib['term']      
        except:
            pass 
        try:
            aspect_category_list = []
            for i, as_cat in enumerate(node[2].iter('aspectCategory')):
                inner_dict['category'+str(i)]=as_cat.attrib['category']      
        except:
            aspect_category_list = []
            for i, as_cat in enumerate(node[1].iter('aspectCategory')):
    #             aspect_category_list.append(as_cat.attrib['category'])
                inner_dict['category'+str(i)]=as_cat.attrib['category']        
        outer_dict[str(n)]=inner_dict 
    df = pd.DataFrame.from_dict(outer_dict, orient = 'index')
    return df




def stem_series(df_series):
    '''
    takes a df series of texts and returns a porter stemmed version of the text 
    to be appended to the dataframe:
        
    '''
    test_text_list = list (df_series)
    stemmer = nltk.PorterStemmer()
    stemmed_texts = []
    for text in test_text_list:
        text_list = text.split(' ')
        stemmed_list = []
        for word in text_list:
            new_word = re.sub('\W', '', word)
            stemmed_list.append(stemmer.stem(new_word))
        stemmed_texts.append(' '.join(stemmed_list))
    return stemmed_texts

def load_lexicon_features (yelp_folder):
    
    yelp_folder = '../../data/yelp_lexicon'
    ambience = '/ambience-unigrams-pmilexicon.txt'
    anecdotes = '/anecdotes-unigrams-pmilexicon.txt'
    food = '/food-unigrams-pmilexicon.txt'
    price = '/price-unigrams-pmilexicon.txt'
    service = '/service-unigrams-pmilexicon.txt'

    list_of_categories = [ambience, anecdotes, food, price, service]
    list_of_dicts = []
    for category in list_of_categories:
        with open (yelp_folder+category, 'r', encoding = 'latin1') as infile:
            f = infile.readlines()

        my_dict = {line.split('\t')[0]: float(line.split('\t')[1]) for line in f}
        list_of_dicts.append(my_dict)    
    dict_list = {'ambience':list_of_dicts[0], 'anecdotes':list_of_dicts[1] , 'food':list_of_dicts[2],
                  'price':list_of_dicts[3], 'service': list_of_dicts[4]} 
    return dict_list

def text_to_score (dataframe_series, dict_list):
    '''takes df_series of texts and list of Yelp lexicon formatted as dicts
    returns a sum of Yelp scores for each of the 5 categories for each text
    
    '''
    temp_list=[]
    temp_score_food = 0
    temp_score_service = 0
    temp_score_anecdotes = 0
    temp_score_ambience = 0
    temp_score_price = 0
    for text in dataframe_series.tolist():

        for key, value in dict_list.items():
            for word in text.split():
                if key == 'ambience':
                    try:
                        temp_score_ambience += value[word.lower()]
                    except:
                        pass
                elif key == 'anecdotes':
                    try:
                        temp_score_anecdotes += value[word.lower()]
                    except:
                        pass
                elif key == 'food':
                    try:
                        temp_score_food += value[word.lower()]
                    except:
                        pass
                elif key == 'price':
                    try:
                        temp_score_price += value[word.lower()]
                    except:
                        pass
                else:
                    try:
                        temp_score_service += value[word.lower()]
                    except:
                        pass
                    
        temp_list.append((temp_score_ambience, temp_score_anecdotes, temp_score_food, temp_score_price, temp_score_service))
        temp_score_food = 0
        temp_score_service = 0
        temp_score_anecdotes = 0
        temp_score_ambience = 0
        temp_score_price = 0
    df_new = pd.DataFrame(temp_list, columns =['ambience', 'anecdotes', 'food', 'price', 'service']) 
    return df_new

def concat_dfs (df_test, df_new_test, df_train, df_new_train):
    """concatenates scores with associated reviews in df_train and df_test

    Args:
        df_test (dataframe): df_test reviews and text; aspect terms; aspect categories 
        df_new_test (dataframe): scores for 5 categories per review
        df_train (dataframe): df_train reviews and text; aspect terms; aspect categories 
        df_new_train (dataframe): scores for 5 categories per review

    Returns:
        [dataframe]: scores concatenated as new columns
    """
    df_test.reset_index(inplace = True, drop= True)
    df_new_test.reset_index(inplace = True, drop= True)
    df_train.reset_index(inplace = True, drop= True)
    df_new_train.reset_index(inplace = True, drop= True)
    df_test = pd.concat([df_test, df_new_test], axis=1)
    df_train = pd.concat([df_train, df_new_train], axis=1)
    
    return df_test, df_train

def add_brown_clusters (path):
    """transforms brown cluster file into a dict of binary keys and word cluster values

    Args:
        path (string): path to brown cluster tweets

    Returns:
        dict: keys are binaries denoting clusters, values are words in clusters
    """
    with open (path, 'r', encoding='utf-8') as infile:
        f = infile.readlines()
    split_f = [line.split('\t')[:2] for line in f]
    cluster_dict = {}
    for ls in split_f:
        if ls[0] in cluster_dict:
            cluster_dict[ls[0]].append(ls[1])
        else:
            cluster_dict[ls[0]]=[ls[1]]
    return cluster_dict

def one_hotter (dictionary, text):
    '''takes a dictionary of clusters and a text returns a one hot encoded vector 
    where each text is represented by a one hot of clusters.
    
    '''
    cluster_keys = sorted(dictionary.keys())
    one_hot = [0]*len(cluster_keys)

    for word in text.split():
        for k, v in dictionary.items():
            if word in v:
                i = cluster_keys.index(k)
                one_hot[i]+=1
    return one_hot

def convert_to_array (one_hot_list):
    """ converts list into array
    """
    return np.array(one_hot_list)
    

def n_gram_builder (df_train, df_test, training_array_clusters, test_array_clusters):
    """constructs n-grams and char n-gram vectors and concatenates with stem and cluster information

    Args:
        df_train (dataframe): CoNLL formatted SemEval 2014 restaurant dataset
        df_test (dataframe): CoNLL formatted SemEval 2014 restaurant dataset
        training_array_clusters (vector): one hot represenation of texts according to occurence of words in brown clusters
        test_array_clusters (vector): one hot represenation of texts according to occurence of words in brown clusters

    Returns:
        2 vectors: training vector and test vector encoded for entry into SVM 
    """
    
    
    cv = CountVectorizer(min_df=2, ngram_range=(1, 2), stop_words='english')
    count_vector=cv.fit_transform(df_train.text.tolist())
    test_vector = cv.transform(df_test.text.tolist())
    X = count_vector
    
    cv_char = CountVectorizer (analyzer = 'char_wb', ngram_range = (4,4), stop_words='english' )
    count_vector_char=cv_char.fit_transform(df_train.text.tolist())
    test_vector_char = cv_char.transform(df_test.text.tolist())
    X_char = count_vector_char
    
    cv_stem = CountVectorizer(min_df=2, ngram_range=(1, 2), stop_words='english')
    count_vector_stem=cv_stem.fit_transform(df_train.stemmed_text.tolist())
    test_vector_stem = cv_stem.transform(df_test.stemmed_text.tolist())
    X_stem = count_vector_stem
    
    df_train_list = list(zip(df_train.ambience,df_train.anecdotes, df_train.food, df_train.price, df_train.service))
    df_test_list = list(zip(df_test.ambience,df_test.anecdotes, df_test.food, df_test.price, df_test.service))
    
    training_array = np.array(df_train_list)
    test_array = np.array(df_test_list)
    
    X_train = np.concatenate((X.todense(), X_char.todense(), X_stem.todense(), training_array, training_array_clusters), axis=1)
    y_train = np.concatenate((test_vector.todense(), test_vector_char.todense(),test_vector_stem.todense(), test_array, test_array_clusters), axis=1)
    
    return X_train, y_train

def populate_df_with_binary_cols_per_label (df_test, df_train):
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

def predict (X_train, y_train, df_train, df_test):
    """prints the classification report for Xu et al.'s algorithm as implemented

    Args:
        X_train (vector): data encoded for entry into SVM
        y_train (vector):  data encoded for entry into SVM 
        df_train (dataframe): CoNLL formatted SemEval 2014 training dataset
        df_test (dataframe): CoNLL formatted SemEval 2014 gold dataset
    """
    svm_var = LinearSVC(random_state=4, tol=1e-5, max_iter=1500)
    
    svm_var.fit(X_train, df_train.cat_ambience.tolist())
    predictions_amb = list(svm_var.predict(y_train))  
    target_names = [1, 0]
    print('ambience', '\n', classification_report(df_test['cat_ambience'], predictions_amb))
    
    svm_var.fit(X_train, df_train['cat_anecdotes/miscellaneous'].tolist())
    predictions_anec = list(svm_var.predict(y_train))  
    target_names = [1, 0]
    print('anecdotes/misc', '\n',classification_report(df_test['cat_anecdotes/miscellaneous'], predictions_anec))
    
    svm_var.fit(X_train, df_train['cat_food'].tolist())
    predictions_food = list(svm_var.predict(y_train))  
    target_names = [1, 0]
    print('food', '\n',classification_report(df_test['cat_food'], predictions_food))
    
    svm_var.fit(X_train, df_train['cat_price'].tolist())
    predictions_price = list(svm_var.predict(y_train))  
    target_names = [1, 0]
    print('price', '\n',classification_report(df_test['cat_price'], predictions_price))
    
    svm_var.fit(X_train, df_train['cat_service'].tolist())
    predictions_service = list(svm_var.predict(y_train))  
    target_names = [1, 0]
    print('service', '\n',classification_report(df_test['cat_service'], predictions_service))
    
    