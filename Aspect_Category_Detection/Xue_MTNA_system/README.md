# README
System is based on Xue et al.'s MTNA-s Aspect Category Detection system

Sole inputs are 200d GloVe embeddings available here: https://nlp.stanford.edu/projects/glove/ 

The system uses 5 binary 1-1 svm classifiers: performance metrics only consider 1's (label), not 0's (label)

Running system is slow - allow at least 20 minutes to run.

The notebook shows a proof of concept for post-processing - although the variance makes this an insconsistent modification

- W. Xue, W. Zhou, T. Li, and Q. Wang.  Mtna:  a neural multi-task model for aspectcategory classification and aspect term extraction on restaurant reviews. InProceed-ings  of  the  Eighth  International  Joint  Conference  on  Natural  Language  Processing(Volume 2:  Short Papers), pages 151–156, 2017.
