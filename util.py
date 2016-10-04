'''
Created on Dec 8, 2015

@author: donghyun
'''
import numpy as np


def eval_RMSE(R, U, V, TS):
    num_user = U.shape[0]
    sub_rmse = np.zeros(num_user)
    TS_count = 0
    for i in xrange(num_user):
        idx_item = TS[i]
        if len(idx_item) == 0:
            continue
        TS_count = TS_count + len(idx_item)
        approx_R_i = U[i].dot(V[idx_item].T)  # approx_R[i, idx_item]
        R_i = R[i]

        sub_rmse[i] = np.square(approx_R_i - R_i).sum()

    rmse = np.sqrt(sub_rmse.sum() / TS_count)

    return rmse


def make_CDL_format(X_base, path):
    max_X = X_base.max(1).toarray()
    for i in xrange(max_X.shape[0]):
        if max_X[i, 0] == 0:
            max_X[i, 0] = 1
    max_X_rep = np.tile(max_X, (1, X_base.shape[1]))
    X_nor = X_base / max_X_rep
    np.savetxt(path + '/mult_nor.dat', X_nor, fmt='%.5f')
