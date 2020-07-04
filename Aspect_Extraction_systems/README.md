# README

Qiu et al.'s Double Propagation algorithm is a frequently cited approach to the task of Aspect Extraction based purely on linguistic information, namely:

Part-of-speech
Dependency relations

A seed lexicon is borrowed from Hu and Liu (2004); there are feature dictionaries, but experiments have shown that the algorithm works equally well with or without this. 

Xu et al.'s Double Embedding with CNN approach uses a custom trained in domain 100 dim embedding concatentated to a 300 dim general embedding (originally using GloVe vectors but implemented with Word2vec trained on Google news content. It's fast and comes close to SOTA systems for both precision and recall. Output is relatively consistent. 
