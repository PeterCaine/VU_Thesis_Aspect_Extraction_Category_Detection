from utils import pickle_loader, integer_encoder, glove_loader,\
    substitue_embeddings, populate_df_with_binary_cols_per_label, \
        run_mtna, predict
def main():
    
    path_to_lemma_train = '../../data/train_for_bi_lstm.pkl'
    path_to_lemma_test = '../../data/gold_for_bi_lstm.pkl'

    df_train = pickle_loader(path_to_lemma_train)
    df_test = pickle_loader(path_to_lemma_test)

    padded_docs, padded_test_docs, t, vocab_size = integer_encoder(df_train, df_test)

    prompt = "Please enter path to GloVe 200d embeddings: "
    path_to_glove = input(prompt)
    embeddings_index = glove_loader (path_to_glove)
    embedding_matrix = substitue_embeddings(embeddings_index,t, vocab_size)

    df_train, df_test = populate_df_with_binary_cols_per_label(df_train, df_test)

    model = run_mtna(embedding_matrix, vocab_size)
    predict(model, padded_docs, df_train)


if __name__ == "__main__":
    main()