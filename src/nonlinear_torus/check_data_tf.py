import matplotlib.pyplot as plt
import numpy as np
import deepxde as dde
from deepxde.backend import tf
from  nonlinear_torus.dm_utils_tf import DiffusionMap
import scipy

dname = "../../data/data_nonlinear/"

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

def compute_res(u, ind_k):
    X = np.loadtxt(f"{dname}Xt_train.txt", delimiter=",")
    kappa = np.loadtxt(f"{dname}kappast_train.txt", delimiter=",")
    dm = DiffusionMap(X, kappa)
    kappa = kappa[ind_k]
    L_mat = dm.get_matrix(kappa)
    u = u[ind_k].reshape((-1,1))
    f = 3/2 * u**2 + 2 * kappa.reshape((-1,1)) * u - 1/2 * kappa.reshape((-1,1))**2 + u
    y = u
    L_mat_current = convert_sparse_matrix_to_sparse_tensor(L_mat)
    y  = np.float32(y )
    kappa  = np.float32(kappa)
    indices = L_mat_current.indices
    values = L_mat_current.values
    dense_shape = L_mat_current.dense_shape
    kappa_reshaped = tf.reshape(kappa, (-1, 1))
    neg_sparse_mat = tf.SparseTensor(indices=indices, values=-values, dense_shape=dense_shape)
    f = 3/2 * tf.square(y) + 2 * kappa_reshaped * y - 1/2 * tf.square(kappa_reshaped) + y 
    eye_sparse = tf.sparse.eye(2500, dtype=tf.float32)
    sparse_mat = tf.sparse.add(neg_sparse_mat, eye_sparse)
    res = tf.sparse.sparse_dense_matmul(sparse_mat, y) - f
    res = res.eval(session=tf.compat.v1.Session())
    res_err = dde.metrics.mean_squared_error(np.zeros((2500, 1)), res.reshape((2500, 1)))
    return res_err

if __name__ == "__main__":
    u = np.loadtxt(f"{dname}/us_train.txt", delimiter=",")
    res = compute_res(u,10)
    print("PDE residual error on training data:", res)