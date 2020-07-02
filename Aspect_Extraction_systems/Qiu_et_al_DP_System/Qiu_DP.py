
from utils import initialise_lexicon, test_set, algorithm, post_process, predict


##########################################################################

def main ():
    path_pos = '../../data/opinion-lexicon-English/positive-words.txt'
    path_neg = '../../data/opinion-lexicon-English/negative-words.txt'
    path_to_testset = '../../data/restaurants_gold.tsv'
    #load in test set data
    prompt = "Please enter filepath to CoNLL formatted SemEval 2014 restaurant test set: "
    input_path_to_testset = input(prompt)
    if input_path_to_testset == '':
        pass
    else:
        path_to_testset = input_path_to_testset
    df_test, df_gold_lists = test_set(path_to_testset)

    prompt = "Please enter filepath to positive seed lexicon: "
    input_path_pos = input(prompt)
    if input_path_pos == '':
        pass
    else:
        path_pos = input_path_pos
    prompt = "Please enter filepath to negative seed lexicon: "
    input_path_neg = input(prompt)
    if input_path_neg == '':
        pass
    else:
        path_neg = input_path_neg
    # populate seed opinion set and create aspect set
    O, F = initialise_lexicon(path_pos, path_neg)
    # run DP algorithm
    features_out_dict = algorithm (df_gold_lists, F, O)
    # convert phrase terms and prepare for prediction
    system_predictions = post_process(features_out_dict, df_test)
    # predict
    predict(df_test, system_predictions)

    
if __name__ == "__main__":
    main()

