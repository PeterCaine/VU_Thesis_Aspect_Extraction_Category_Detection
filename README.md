# README
This repo contains the code used for my thesis project at the VU (Vrije Universiteit - Amsterdam).

Aspect Based Sentiment Analysis is an approach to sentiment analysis which analyses texts at the word/token level in order to determine whether or not it is or forms part of a phrase which is the target of a subjective expression. According to the parameters laid down in SemEval 2014, there are 4 subtasks:
- aspect extraction (AE)
- aspect polarity classification
- aspect category detection (ACD)
- category polarity classification

This thesis reviews studies involving both approaches and selected systems from both approaches to re-implement with the aim of evaluating the output to uncover qualitative differences.
Minimally modified versions of 4 existing systems discovered in the literature (2 AE systems; 2 ACD systems) were implemented in order that output on an existing dataset could be inspected:

2 AE systems: 
- a system based on Qiu et al.'s Double Propogation algorithm (2011)
- a system based on Xu et al.'s Double Embedding CNN system (2018)

2 ACD systems:
- a system based on Kirtichenko et al.'s SemEval 2014 entry (2014)
- a system based on Xue et al.'s Multi Task Neural Approach system (2019)

### Resources:
The code and links to resources used are here for reference:
Most of the code can be run from the terminal. Much of the data required is (temporarily) available in the data folder. This will be taken offline after review. 
Links to the original data include:
- SemEval 2014 task and description: http://alt.qcri.org/semeval2014/task4/
- SemEval 2014 datasets: http://www.metashare.org/ (direct links from above)
- 1000 Brown clusters (trained on tweets) available at: http://www.cs.cmu.edu/~ark/TweetNLP/
- Yelp Restaurant Word–Aspect Association Lexicon available at: http://www.saifmohammad.com/WebPages/lexicons.html
- Hu and Liu (2004) seed opinion dataset: https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html

## References:
- S. Kiritchenko,  X. Zhu,  C. Cherry,  and S. Mohammad.  Nrc-canada-2014:  Detecting aspects and sentiment in customer reviews. In Proceedings  of  the  8th  international workshop on semantic evaluation (SemEval 2014), pages 437–442, 2014
- G. Qiu,  B. Liu,  J. Bu,  and C. Chen.  Opinion word expansion and target extraction through double propagation. Computational linguistics, 37(1):9–27, 2011
- H.  Xu,  B.  Liu,  L.  Shu,  and  P.  S.  Yu. Double  embeddings  and  cnn-based  sequence labeling for aspect extraction.arXiv preprint arXiv:1805.04601, 2018
- W. Xue, W. Zhou, T. Li, and Q. Wang.  Mtna:  a neural multi-task model for aspect category classification and aspect term extraction on restaurant reviews. In Proceed-ings of the Eighth  International Joint Conference  on  Natural  Language Processing (Volume 2: Short Papers), pages 151–156, 2017.

## Thesis
A copy of the written part of the thesis is available <a href="https://github.com/PeterCaine/VU_Thesis_Aspect_Extraction_Category_Detection/blob/master/Caine_2671676_Thesis.pdf"> here </a>
