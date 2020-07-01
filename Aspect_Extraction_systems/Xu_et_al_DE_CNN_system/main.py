from utils import load_test_train, load_embeddings, training_data, test_data, run_cnn


def main (verbose = False):
     
    path_train = '../../data/restaurants_train.tsv'
    path_test = '../../data/restaurants_gold.tsv'
    
    prompt = 'please input path to SemEval 2014 CoNLL formatted training data: '
    input_path_train = input(prompt)
    if input_path_train == '':
        pass
    else:
        path_train = input_path_train
    prompt = 'please input path to SemEval 2014 CoNLL formatted test data: '
    input_path_test = input(prompt)
    if input_path_test == '':
        pass
    else:
        path_test = input_path_test

    df_train, df_test = load_test_train(path_train, path_test)
    
    embed_prompt = "please enter path to 300-dimensional Word2vec Google News vectors: "
    w2v_path = input (embed_prompt)    
    xu_embed_prompt = "please enter path to Xu et al.'s 100-d restaurant domain vectors: "
    xuv_path = input (xu_embed_prompt)
    #load embeddings
    w2v, xuv = load_embeddings(w2v_path, xuv_path)
    
    X_train, X_valid, y_train, y_valid = training_data (w2v, xuv, df_train['word'], df_train['label'])
    X_test, label_index = test_data (w2v, xuv, df_test['word'], df_test['label'])
    
    # run DE_CNN NN
    batch_size = 128
    embedding_dims = X_train.shape[2]
    kernel_size = 5
    epochs = 200
    model = run_cnn(X_train, y_train, X_valid, y_valid, embedding_dims = embedding_dims,
                    batch_size = batch_size, kernel_size = kernel_size, epochs = 200, 
                    verbose = verbose)
    #predict
    predict(model, df_test, label_index)

if __name__ == "__main__":
    main(verbose = True)