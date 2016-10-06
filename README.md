#Convolutional Matrix Factorization for Document Context-Aware Recommendation (RecSys 2016)
_**Donghyun Kim**_, Chanyoung Park, Jinoh Oh, Seungyong Lee, Hwanjo Yu

###Overview
> Sparseness of user-to-item rating data is one of the major factors that deteriorate the quality of recommender system. To handle the sparsity problem, several recommendation techniques have been proposed that additionally consider auxiliary information to improve rating prediction accuracy. In particular, when rating data is sparse, document modeling-based approaches have improved the accuracy by additionally utilizing textual data such as reviews, abstracts, or synopses. However, due to the inherent limitation of the bag-of-words model, they have difficulties in effectively utilizing contextual information of the documents, which leads to shallow understanding of the documents. This paper proposes a novel context-aware recommendation model, convolutional matrix factorization (ConvMF) that integrates convolutional neural network (CNN) into probabilistic matrix factorization (PMF). Consequently, ConvMF captures contextual information of documents and further enhances the rating prediction accuracy.


###Requirements

- Python 2.7
- Keras 0.3.3
 - [See installation instructions] (https://github.com/cartopy/keras-0.3.3)

###How to Run

Note: Run `python <install_path>/run.py -h` in bash shell. You will see how to configure several parameters for ConvMF

###Configuration
You can evaluate our methods with different settings in terms of the size of dimension, the value of hyperparameter, the number of convolutional kernal, and etc. Below is descriptions of all the configurable parameters and their defaults:
Parameter|Description|Default
---|---|---
`-h`, `--help`|show this help message|{}
`-c <bool>`, `--do_preprocess <bool>`|preprocess raw data for ConvMF|False
`-r <path>`, `--raw_rating <path>`|Set path to raw rating data. data format should be `user id::item id::rating`

