from utils import xml_category_df, stem_series, load_lexicon_features,\
    text_to_score, concat_dfs, add_brown_clusters ,one_hotter, \
        n_gram_builder, convert_to_array, \
            populate_df_with_binary_cols_per_label, predict

def main ():
    
    xml_file_test = '../../data/Restaurants_Test_Gold.xml'
    xml_file_train = '../../data/Restaurants_Train.xml'
    yelp_path = '../../data/yelp_lexicon'
    brown_path = '../../data/brown_clusters_tweets/brown_clusters_tweets'
    
    prompt = "Please enter path to Restaurants_Test_Gold.xml file: "
    xml_file_test_input = input(prompt)
    if xml_file_test_input == '':
        pass
    else:
        xml_file_test = xml_file_test_input
    prompt = "Please enter path to Restaurants_Train.xml file: "
    xml_file_train_input = input(prompt)
    if xml_file_train_input == '':
        pass
    else:
        xml_file_train = xml_file_train_input
    prompt = "Please enter path to yelp lexicon features folder: "
    yelp_path_input = input(prompt)
    if yelp_path_input == '':
        pass
    else:
        yelp_path = yelp_path_input
    prompt = "Please enter path to brown_clusters_tweets file: "
    brown_path_input = input(prompt)
    if brown_path_input == '':
        pass
    else:
         brown_path = brown_path_input
    df_test = xml_category_df (xml_file_test)
    df_train = xml_category_df (xml_file_train)

    df_train['stemmed_text'] = stem_series(df_train.text)
    df_test['stemmed_text'] = stem_series(df_test.text)

    dict_list = load_lexicon_features(yelp_path)

    df_new_train = text_to_score (df_train.text, dict_list)
    df_new_test = text_to_score (df_test.text, dict_list)
    df_test, df_train = concat_dfs(df_test, df_new_test, df_train, df_new_train)
    test_texts = list (df_test.text)
    train_texts = list (df_train.text)
    cluster_dict = add_brown_clusters(brown_path)
    one_hot_test_list = [one_hotter(cluster_dict, text) for text in test_texts]
    one_hot_train_list = [one_hotter(cluster_dict, train) for train in train_texts]
    training_array_clusters = convert_to_array(one_hot_train_list)
    test_array_clusters = convert_to_array(one_hot_test_list)
    X_train, y_train = n_gram_builder(df_train, df_test, training_array_clusters, test_array_clusters)
    df_train, df_test = populate_df_with_binary_cols_per_label(df_test, df_train)
    predict(X_train, y_train, df_train, df_test)


if __name__ == "__main__":
    main()