# README

Based on Kiritchenko et al.'s (2014) system 

uses features:
unigrams-grams, binary n-grams, char-grams (n=4), stemmed words (Porter stermmer), Lexicon features (Yelp Restaurant Word–Aspect Association Lexicon), 
Brown clusters (trained on tweets).

The system uses 5 binary 1-1 svm classifiers: 
performance metrics only consider 1's (label), not 0's (label)


 - Yelp Restaurant Word–Aspect Association Lexicon available at:http://www.saifmohammad.com/WebPages/lexicons.html
 - Brown clusters tweets available here: http://www.cs.cmu.edu/~ark/TweetNLP/ 
 
 - S. Kiritchenko,  X. Zhu,  C. Cherry,  and S. Mohammad.  Nrc-canada-2014:  Detectingaspects and sentiment in customer reviews.  InProceedings  of  the  8th  internationalworkshop on semantic evaluation (SemEval 2014), pages 437–442, 2014.
