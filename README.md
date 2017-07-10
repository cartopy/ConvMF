# Convolutional Matrix Factorization (ConvMF)

### Overview
> Sparseness of user-to-item rating data is one of the major factors that deteriorate the quality of recommender system. To handle the sparsity problem, several recommendation techniques have been proposed that additionally consider auxiliary information to improve rating prediction accuracy. In particular, when rating data is sparse, document modeling-based approaches have improved the accuracy by additionally utilizing textual data such as reviews, abstracts, or synopses. However, due to the inherent limitation of the bag-of-words model, they have difficulties in effectively utilizing contextual information of the documents, which leads to shallow understanding of the documents. This paper proposes a novel context-aware recommendation model, convolutional matrix factorization (ConvMF) that integrates convolutional neural network (CNN) into probabilistic matrix factorization (PMF). Consequently, ConvMF captures contextual information of documents and further enhances the rating prediction accuracy.

### Paper
- Convolutional Matrix Factorization for Document Context-Aware Recommendation (*RecSys 2016*)
  - <a href="http://dm.postech.ac.kr/~cartopy" target="_blank">_**Donghyun Kim**_</a>, Chanyoung Park, Jinoh Oh, Seungyong Lee, Hwanjo Yu
- Deep Hybrid Recommender Systems via Exploiting Document Context and Statistics of Items (*Information Sciences (SCI)*)
   - <a href="http://dm.postech.ac.kr/~cartopy" target="_blank">_**Donghyun Kim**_</a>, Chanyoung Park, Jinoh Oh, Hwanjo Yu

### Requirements

- Python 2.7
- Keras 0.3.3
 - <a href="https://github.com/cartopy/keras-0.3.3" target="_blank">See installation instructions</a>

### How to Run

Note: Run `python <install_path>/run.py -h` in bash shell. You will see how to configure several parameters for ConvMF

### Configuration
You can evaluate our model with different settings in terms of the size of dimension, the value of hyperparameter, the number of convolutional kernal, and etc. Below is a description of all the configurable parameters and their defaults:

Parameter | Default
---       | ---
`-h`, `--help` | {}
`-c <bool>`, `--do_preprocess <bool>` | `False`
`-r <path>`, `--raw_rating_data_path <path>` | {}
`-i <path>`, `--raw_item_document_data_path <path>`| {}
`-m <integer>`, `--min_rating <integer>` | {}
`-l <integer>`, `--max_length_document <integer>` | 300
`-f <float>`, `--max_df <float>` | 0.5
`-s <integer>`, `--vocab_size <integer>` | 8000
`-t <float>`, `--split_ratio <float>` | 0.2
`-d <path>`, `--data_path <path>` | {}
`-a <path>`, `--aux_path <path>` | {}
`-o <path>`, `--res_dir <path>` | {}
`-e <integer>`, `--emb_dim <integer>` | 200
`-p <path>`, `--pretrain_w2v <path>` | {}
`-g <bool>`, `--give_item_weight <bool>` | `True`
`-k <integer>`, `--dimension <integer>` | 50
`-u <float>`, `--lambda_u <float>` | {}
`-v <float>`, `--lambda_v <float>` | {}
`-n <integer>`, `--max_iter <integer>` | 200
`-w <integer>`, `--num_kernel_per_ws` | 100

1. `do_preprocess`: `True` or `False` in order to preprocess raw data for ConvMF.
2. `raw_rating_data_path`: path to a raw rating data path. The data format should be `user id::item id::rating`.
3. `min_rating`: users who have less than `min_rating` ratings will be removed.
4. `max_length_document`: the maximum length of document of each item.
5. `max_df`: threshold to ignore terms that have a document frequency higher than the given value. i.e. for removing corpus-stop words.
6. `vocab_size`: size of vocabulary.
7. `split_ratio`: 1-ratio, ratio/2 and ratio/2 of the entire dataset will be constructed as training, valid and test set, respectively.
8. `data_path`: path to training, valid and test datasets.
9. `aux_path`: path to R, D_all sets that are generated during the preprocessing step.
10. `res_dir`: path to ConvMF's result
11. `emb_dim`: the size of latent dimension for word vectors.
12. `pretrain_w2v`: path to pretrained word embedding model to initialize word vectors.
13. `give_item_weight` : `True` or `False` to give item weight for R-ConvMF.
14. `dimension`: the size of latent dimension for users and items.
15. `lambda_u`: parameter of user regularizer.
16. `lambda_v`: parameter of item regularizer.
17. `max_iter`: the maximum number of iteration.
18. `num_kernel_per_ws`: the number of kernels per window size for CNN module.

