import nltk
import pandas as pd
from sklearn.metrics import classification_report


def initialise_lexicon (path_pos, path_neg):
    """takes paths to two lexicon .txt files (one positive opinion lexicon and one negative) 
    returns a set of seed opinion words for use in the DP algorithm. 

    Args:
        path_pos (string): path to positive lexicon .txt file
        path_neg (string): path to negative lexicon .txt file
    """
    with open (path_pos, 'r', encoding = 'utf-8') as infile:
        p_f = infile.readlines()
    pos_O = set([line.strip() for line in p_f])

    with open (path_neg, 'r', encoding = 'utf-8') as infile:
        n_f = infile.readlines()
    neg_O = set([line.strip() for line in n_f])
    O = pos_O|neg_O
    F=set()
    
    return O , F

def test_set (path_to_testset):
    """takes path, returns CoNLL formatted SemEval 2014 testset
    returns tsv as dataframe and list of dataframes: 1 df per review

    Args:
        path_to_testset (string): path to reformatted dataset restaurants_gold.tsv
    """

    # load in test data 2014 - gold
    df_test= pd.read_csv(path_to_testset, sep = '\t')
    gb_gold = df_test.groupby('review_id')
    df_gold_lists = [gb_gold.get_group(x) for x in gb_gold.groups]
    
    return df_test, df_gold_lists

def populate (df_lists,n):
    '''creates variables from lists of dataframes for each rule.
    '''
    word = df_lists[n].word.tolist()
    deps = df_lists[n].deprel.tolist()
    heads = df_lists[n]['head'].tolist()
    lem_index = df_lists[n].lemma_index.tolist()
    lemmas = df_lists[n].lemma.tolist()
    upos = df_lists[n].upos.tolist()

    return word, deps, heads, lem_index, lemmas, upos

def r1 (df_lists, O, F_dict):
    '''extracts new feature nouns (aspect targets)
    takes a lists extracted from dataframe for dependencies, heads, pos, and lemmas
    outputs set of extracted features (aspect terms) identified by deprel to known opinion adjectives
    
    '''
    set_r1_f = set()
    for n in range(len(df_lists)):
        #this is our populate function from above
        word, deps, heads, lem_index, lemmas, upos = populate (df_lists,n)
        for i, relation in enumerate(deps):
            #the head col points to an index which does not start at 0 (that is root in deprel language) 
            #so 1 must be subtracted to get the index of the thing it's pointing to
            head = heads[i]-1
            # RM is our list of dependency relations  # O is our opinion lexicon
            if lemmas[i] in O and upos[i] == 'ADJ':
                target_head = i+1
                if upos[head] == 'NOUN':
                    dict_targets = lem_index[head].lower()
                    set_targets = lemmas[head].lower()
                    set_r1_f.add(set_targets)
                    if str(n) in F_dict:
                        F_dict[str(n)].append(dict_targets)    
                    else:
                        F_dict[str(n)]=[dict_targets] 
    return set_r1_f

def r2 (df_lists, RM, O, F_dict):
    '''extracts new features (aspect targets)
    takes a lists extracted from dataframe for dependencies, heads, upos, and lemmas
    outputs set of extracted features (aspect terms) identified by sharing a head with the head of known opinion adjectives

    '''
    set_r2_f = set()
    for n in range(len(df_lists)):
        word, deps, heads, lem_index, lemmas, upos = populate (df_lists,n)
        for i, relation in enumerate(deps):        
            if lemmas [i] in O and upos[i] == 'ADJ' and relation in RM and upos[i] == 'ADJ':
                target_head = heads[i]
                
                for x, rel in enumerate(deps):
                    if x !=i and upos[x] =='NOUN' and rel in RM and heads[x] == target_head:
                        dict_targets = lem_index[x].lower() 
                        set_targets = lemmas[x].lower()           
                        set_r2_f.add(set_targets)
                        if str(n) in F_dict:
                            F_dict[str(n)].append(dict_targets)    
                            
                        else:
                            F_dict[str(n)]=[dict_targets]                      
                            
    return set_r2_f 

def r3 (df_lists, RM, F):
    '''extracts new opinion words
    takes a lists extracted from dataframe for dependencies, heads, pos, and lemmas
    outputs set of opinion words identified by having a dependency relation with a known feature

    '''
    # extract targets
    set_r3_o = set()
    for n in range(len(df_lists)):
        word, deps, heads, lem_index, lemmas, upos = populate (df_lists,n)
        for i, relation in enumerate(deps):
            #the head col points to an index which does not start at 0 (that would be root) so 1 must be subtracted to get the index of the thing it's pointing to
            head = heads[i]-1
            # RM is our list of dependency relations
            # F is our list of f
            if relation in RM and lemmas[head] in F and upos[i] == 'ADJ':  
                set_r3_o.add(lemmas[i].lower())
    return set_r3_o

def r4 (df_lists, RM, F):
    '''extracts new opinion words
    takes a lists extracted from dataframe for dependencies, heads, pos, and lemmas
    outputs set of opinion identified by sharing a head with the head of known target feature

    '''
    set_r4_o = set()
    
    for n in range(len(df_lists)):
        word, deps, heads, lem_index, lemmas, upos = populate (df_lists,n)
        for i, relation in enumerate(deps):
#             head = heads[i]-1
            if relation in RM and upos[i] == 'NOUN' and lemmas[i] in F:
                target_head = heads[i]
#                 target_subject = lemma_sents[i]
                for x, rel in enumerate(deps):
                    if x != i and rel in RM and upos[x] == 'ADJ' and heads[x]==target_head:
                        set_r4_o.add(lemmas[x])
                    
    return set_r4_o

def r5(df_lists, F_dict, F):
    '''extracts new features (aspect targets)
    takes a lists extracted from dataframe for dependencies, heads, pos, and lemmas
    uses known list of features to extract other features with a conj relation
    '''
    set_r5_f = set()
    
    for n in range(len(df_lists)):
        word, deps, heads, lem_index, lemmas, upos = populate (df_lists,n)
        for i, relation in enumerate(deps):      
        #the head col points to an index which does not start at 0 so 1 must be subtracted to get the index of the thing it's pointing to
            head = heads[i]-1
            if relation == 'conj' and upos[i] == 'NOUN' and lemmas[head] in F:
                set_targets = lemmas[i].lower()
                dict_targets = lem_index[i].lower()
                
                set_r5_f.add(set_targets)
                if str(n) in F_dict:
                    F_dict[str(n)].append(dict_targets)    
                else:
                    F_dict[str(n)]=[dict_targets]
               
    return set_r5_f

def r6(df_lists, RM, F, F_dict):
    '''extracts new features (aspect targets)
    takes a lists extracted from dataframe for dependencies, heads, pos, and lemmas
    uses known list of features to extract other features with a shared head and shared dependency relation
    '''  
    set_r6_f = set()

    for n in range(len(df_lists)):
        word, deps, heads, lem_index, lemmas, upos = populate (df_lists,n)
        for i, relation in enumerate(deps):  
        #the head col points to an index which does not start at 0 so 1 must be subtracted to get the index of the thing it's pointing to
            head = heads[i]-1  
            if relation in RM and upos[i] == 'NOUN' and lemmas[i] in F:          
                target_head = head
                for x, rel in enumerate(deps):
                    if x != i and upos[x] == 'NOUN' and target_head == heads[x] and (rel == relation or (rel in ['nsubj', 'csubj', 'xsubj','dobj', 'iobj'] and relation in ['nsubj', 'csubj', 'xsubj','dobj', 'iobj'])):
                        dict_targets = lem_index[x]
                        set_targets = lemmas[x]
                        set_r6_f.add(set_targets)
                        
                        if str(n) in F_dict:
                            F_dict[str(n)].append(dict_targets)    
                        else:
                            
                            F_dict[str(n)]=[dict_targets]           
    return set_r6_f

def r7 (df_lists, O):
    '''extracts new opinion words 
    takes a lists extracted from dataframe for dependencies, heads, pos, and lemmas
    outputs set of extracted opinion words which share a conj relation with known opinion adjectives

    '''
    set_r7_o = set()
    for n in range(len(df_lists)):
        word, deps, heads, lem_index, lemmas, upos = populate (df_lists,n)
        for i, relation in enumerate(deps):
            head = heads[i]-1
            if relation == 'conj' and upos[i] == 'ADJ' and lemmas[i] in O:
                if upos[head] == 'ADJ':
                    set_r7_o.add(lemmas[head].lower())
                    
    return set_r7_o

def r8 (df_lists, RM, O):
    '''extracts new opinion words 
    takes a lists extracted from dataframe for dependencies, heads, pos, and lemmas
    outputs set of extracted opinion words which share a head with a known opinion adjectives
    
    '''
    set_r8_o = set()
    for n in range(len(df_lists)):
        word, deps, heads, lem_index, lemmas, upos = populate (df_lists,n)
        for i, relation in enumerate(deps):
            #the head col points to an index which does not start at 0 so 1 must be subtracted to get the index of the thing it's pointing to
            head = heads[i]-1
            if lemmas[i] in O and upos[i] == 'ADJ' and relation in RM: 
                target_head = heads[i]
                for x, rel in enumerate(deps):
                    if x != i and upos[x] == 'ADJ' and heads[x]==target_head and (rel == relation or (rel in ['nsubj', 'csubj', 'xsubj','dobj', 'iobj'] and relation in ['nsubj', 'csubj', 'xsubj','dobj', 'iobj'])) :
                        set_r8_o.add(lemmas[x].lower())       
    return set_r8_o

def r9 (df_lists, F_dict):
    
    '''extracts new feature nouns (aspect targets)
    takes a lists extracted from dataframe for dependencies, heads, pos, and lemmas
    outputs set of extracted features (aspect terms) identified by deprel to known opinion adjectives
    
    '''
    set_r9_f = set()
    for n in range(len(df_lists)):
        #this is our populate function from above
        word, deps, heads, lem_index, lemmas, upos = populate (df_lists,n)
        for i, relation in enumerate(deps):
            if relation == 'obj' and upos[i] == 'NOUN':
                dict_targets = lem_index[i].lower()
                set_targets = lemmas[i].lower()
                set_r9_f.add(set_targets)
                if str(n) in F_dict:
                    F_dict[str(n)].append(dict_targets)    
                else:
                    F_dict[str(n)]=[dict_targets]                   
    return set_r9_f

def r10 (df_lists, F_dict):
    
    '''extracts new feature nouns (aspect targets)
    takes a lists extracted from dataframe for dependencies, heads, pos, and lemmas
    outputs set of extracted features (aspect terms) identified by deprel to known opinion adjectives
    
    '''
    set_r10_f = set()
    for n in range(len(df_lists)):
        word, deps, heads, lem_index, lemmas, upos = populate (df_lists,n)
        for i, relation in enumerate(deps):
            head = heads[i]-1
            
            if relation == 'compound' and upos[i] == 'NOUN' and upos[head]=='NOUN':
                dict_targets = lem_index[i].lower() + ' ' + lem_index[head].lower()
                set_targets = lemmas[i].lower() + ' ' + lemmas[head].lower()
                
                set_r10_f.add(set_targets)
                if str(n) in F_dict:
                    F_dict[str(n)].append(dict_targets)    
                else:
                    F_dict[str(n)]=[dict_targets]                   
    return set_r10_f

def r11 (df_lists, O, F_dict):
    """extracts new features (aspects) using noun as head of opinoin

    Args:
        df_lists (list): list of datframes each df represents one review
        text

    Returns:
        set: feature set to be updated during algorithm
    """
    set_rll_f = set()
    for n in range (len(df_lists)):
        word, deps, heads, lem_index, lemmas, upos = populate (df_lists,n)
        for i, relation in enumerate(deps):
            head = heads[i]-1
            if upos[i] == 'NOUN' and lemmas[head] in O:
                dict_targets = lem_index[i].lower()
                set_targets = lemmas[i].lower()
                set_rll_f.add(set_targets)
                if str(n) in F_dict:
                    F_dict[str(n)].append(dict_targets)    
                else:
                    F_dict[str(n)]=[dict_targets]                   
    return set_rll_f

def algorithm (df_lists, F, O):
    '''a function that iterates through the rules in the order given by Qiu et al (2011) until 
    no more features (F) or Opinion words (O) can be found
    '''
    # list of dependency relations as produced by Stanord CoreNLP for DP algorithm
    RM = set(['amod', 'nsubj', 'csubj', 'xsubj', 'dobj', 'iobj', 'conj', 'obl'])
    
    F = set()
    F_dict = {}
    
    len_F = -1
    len_O = 0
    
    #first pass using known opinion words
    while len_F - len(F) !=0 and len_O - len(O) !=0:
        len_F = len(F)
        len_O = len(O)
        print ('F length: ', len(F))
        print ('O length: ', len(O))
        
        #first pass with known opinion words
        rule1_features = r1 (df_lists, O, F_dict) 
        rule2_features = r2 (df_lists, RM, O, F_dict)
        rule9_features = r9 (df_lists, F_dict)
        rule10_features = r10 (df_lists, F_dict)
        rule11_features = r11 (df_lists, O, F_dict)
        rule7_opinion = r7 (df_lists, O)    
        rule8_opinion = r8 (df_lists, RM, O)
        
        #update feature set and opinion set
        F.update(rule1_features|rule2_features|rule9_features|rule10_features|rule11_features)
        O.update(rule7_opinion|rule8_opinion)
        
        #first pass using known target words
        rule5_features = r5 (df_lists, F_dict, F) 
        rule6_features = r6 (df_lists, RM, F, F_dict)
        rule3_opinion = r3 (df_lists, RM, F)    
        rule4_opinion = r4 (df_lists, RM, F)
        #     #update feature set and opinion set
        F.update(rule5_features|rule6_features)
        O.update(rule3_opinion|rule4_opinion)
    
    return F_dict

grammar_dict = {
    
    'n_n': '''
    NP: {<NN.*>}
    N_N: {<NP>+}
    ''',
    'adj_n': '''
    ADJ: {<JJ.*>*} # adjective
    NP: {<JJ.*>* <NN.*>+}
    OT: {<NP> <IN> <DT> <NP>}
    A_N: {<ADJ>* <NP>*}
    '''
}

def extract_grammar(dataframe, grammar, min_len=0):
    '''extracts a pre-determined grammar pattern from a conll formatted corpus
    based on the pattern entered.

    params: dataframe - pandas dataframe which will be split into sentences
    (this avoids having to do FAS look aheads for the whole conll file) -
    increases speed significantly

    params: grammar - a grammar pattern - choose from: [vp, adv_adj,
    v_adj_or_adv, subj_v, n_n, adj_n, adv_v]

    params: min_len: int - minimum length of desired output 0 includes single
    word phrases - 1 includes multi-word phrases

    returns a list of phrases of given grammar pattern of min length indicated

    '''

    grm = grammar_dict[grammar]

    pattern_lookup = {'n_n': 'N_N', 'adj_n': 'A_N'}
    pattern = pattern_lookup[grammar]

    gb = dataframe.groupby('review_id')
    df_lists = [gb.get_group(x) for x in gb.groups]

    output_list = []
    output_dict = {}
    for n, df in enumerate(df_lists):
#         lemmas = [f'{ele}_{i}' for i, ele in enumerate(df.lemma)]
        sentence = list(zip(df.lemma_index, df.xpos))

        
        pattern_list = []
        cp = nltk.RegexpParser(grm)
        result = cp.parse(sentence)
        for subtree in result.subtrees(filter=lambda t: t.label() == pattern):
            if len(subtree) > min_len:
                pattern_list.append(subtree.leaves())

        chunk_list = []
        for element in pattern_list:
            chunk = []
            for ele in element:
                chunk.append(ele[0])
            try:
                joined_chunk = ' '.join(chunk)
                chunk_list.append(joined_chunk)
            except:
                pass
        output_list.extend(chunk_list)
        output_dict[str(n)] = chunk_list
    return output_list, output_dict

def phrase_extractor (df, grammar= 'n_n', min_len = 0, filter_keyword = None):
    '''takes full dataframe and extracts a given grammar pattern, if filter_keyword given
    will only extract phrases with those keywords
    returns a set of all phrases with index
    '''
    #there's a sentence dict here that we are not using
    interim_list, interim_dict = extract_grammar(df, grammar = grammar, min_len=min_len)
    for_output = []
    if filter_keyword:
        for item in interim_list:
            sublist = item.split()    
            for sub in sublist:
                if filter_keyword == sub.split('_')[0]:
                    for_output.append(item)
    else:
        for_output = interim_list
    # we only want multiple words (phrases)        
    output_set = {item for item in for_output if len(item.split())>1}
    output_dict = {}
    for key, value in interim_dict.items():
        output_dict[key] = [v for v in value if len (v.split())>1]
    
    return output_set, output_dict

def F_dict_replace (F_dict, output_dict):
    '''replaces any single words with corresponding phrases if applicable
    
    '''
    # all values are singles in F_dict
    for key, set_o_singles in F_dict.items():
        #need to convert set to list in order to select the value to replace
        list_o_singles = list(set_o_singles)
        for n, item in enumerate(list_o_singles):
            #takes output dict of phrase_extractor
            for k,list_o_multis in output_dict.items():
    #             #checks that the sentence numbers are the same
                if k == key: 
                    for i, multi in enumerate(list_o_multis):
                        #check this is a multi-word
                        if item in multi.split() and len(multi.split())>1:
                            #replace single with multi where applicable
                            list_o_singles[n]=list_o_multis[i]
                            F_dict[key] = list_o_singles
    return F_dict

def word_index_separator (F_dict):
    """ensures no index collisions words in a review were concatenated
    with their index position; this function now removes them

    Args:
        F_dict (dictionary): output of F_dict replace

    Returns:
        dict: dict of words or phrases identifed per review for 
        assigment
    """
    dict_for_assignment = {}
    for key, value in F_dict.items():
        for v in value:
            if len(v.split())==1: 
                word, index = v.split('_')[0], v.split('_')[1]
                tups = (word, index)
                if key in dict_for_assignment:
                    dict_for_assignment[key].append(tups)
                else:
                    dict_for_assignment[key]=[tups]
            else:
                multi_tup = []
                for items in v.split():
                    word, index = items.split('_')
                    tups2 = (word, index)
                    multi_tup.append(tups2)
                if key in dict_for_assignment:
                    dict_for_assignment[key].append(multi_tup)
                else:
                    dict_for_assignment[key]=[multi_tup]
    return dict_for_assignment

def semeval_BIO_label (df, dict_for_assignment):
    """takes a conll formated gold dataframe and dict of aspect terms
    adds these terms as IOB tags to dataframe for evaluation

    Args:
        df ([type]): [description]
        dict_for_assignment ([type]): [description]

    Returns:
        [type]: [description]
    """
    df['system_out'] = 'O'
    for key, value in dict_for_assignment.items():        
        for v in value:
            if type(v) == tuple:
                idx = int(v[1])+1         
                df.system_out[(df.review_id==int(key))&(df.word_id == idx)&(df.lemma == v[0])] = 'B'
            else:
                for i,item in enumerate(v):                
                    if i == 0:
                        indx = int(item[1])+1 
                        df.system_out[(df.review_id==int(key))&(df.word_id == indx)&(df.lemma == item[0])] = 'B'
                        
                    else:
                        indx = int(item[1])+1
                        
                        df.system_out[(df.review_id==int(key))&(df.word_id == indx)&(df.lemma == item[0])] = 'I'
                        
    return df

def post_process (dict_of_single_aspect_terms, df_gold):
    """post processing step to expand single aspect terms to phrases as 
    identified by an nltk chunk parser

    Args:
        dict_of_single_aspect_terms (dict): output of algorithm
        df_gold (dataframe): conll formatted tsv as df

    Returns:
        pandas series: list of aspect phrases
    """
    F_dict = {}
    F_dict = {k:set(v) for k,v in dict_of_single_aspect_terms.items()}
    set_out, dict_out = phrase_extractor (df_gold, grammar= 'n_n', min_len = 0, filter_keyword = None)
    # phrase substitution
    F_Dict = F_dict_replace (F_dict, dict_out)
    # preparing dict for assignment
    dict_for_assignment = word_index_separator (F_dict)
    df_to_csv = semeval_BIO_label(df_gold, dict_for_assignment)
    output_list = df_to_csv.system_out
    return output_list

def predict (df_test, system_predictions):
    """prints the classification report for Qiu et al.'s algorithm as implemented

    Args:
        df_test (dataframe): CoNLL formatted SemEval 2014 gold dataset
        system_predictions (pandas series ): list of aspect phrases after post
        processing
    """

    gold_series = df_test.label.tolist()
    target_names = ['B','I','O']
    print('strict performance report', '\n', classification_report(gold_series, system_predictions, target_names=target_names))
    df_test['predicted'] = system_predictions
    df_test['gold_bi']= [label.replace ('B', 'I') for label in gold_series]
    df_test['predicted_bi']= df_test.predicted.replace({'B': 'I'})
    target_names = ['I','O']
    print('relaxed performance report', '\n', classification_report(df_test.gold_bi, df_test.predicted_bi, target_names=target_names))

