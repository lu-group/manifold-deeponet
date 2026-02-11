import os
import time
import numpy as np
import deepxde as dde

import torch
import torch.nn as nn
import torch.nn.functional as F

from gmls_utils_pi import PDEOperatorRBFCartesianProd, KappaFuncSpace

dde.config.disable_xla_jit()
dde.config.set_default_float("float32")

DNAME = "../../data/data_semi/"
os.makedirs("rbf_res", exist_ok=True)

def load_test_data(fname, m, n):
    if m == 0 and n == 0:
        return None, None
    d = np.load(fname, allow_pickle=True)
    X_test_1, X_test_2, y_test = d["X_test_branch"], d["X_test_trunk"], d["y_test"]
    X_test = (X_test_1[:m, :], X_test_2[:n, :])
    return X_test, y_test[:m, :n]


def load_train_data(fname, m, n):
    if m == 0 and n == 0:
        return None, None
    d = np.load(fname, allow_pickle=True)
    X_train_1, X_train_2, y_train = d["X_train_branch"], d["X_train_trunk"], d["y_train"]
    X_train = (X_train_1[:m, :], X_train_2[:n, :])
    return X_train, y_train[:m, :n]


def load_data():
    d = np.load(f"{DNAME}/semitorus_data_train1_rbf.npz", allow_pickle=True)
    X_train_1, X_train_2, y_train = d["X_train_branch"], d["X_train_trunk"], d["y_train"]
    X_train = (X_train_1, X_train_2)

    d = np.load(f"{DNAME}/semitorus_data_test1_rbf.npz", allow_pickle=True)
    X_test_1, X_test_2, y_test = d["X_test_branch"], d["X_test_trunk"], d["y_test"]
    X_test = (X_test_1, X_test_2)
    return X_train, y_train, X_test, y_test


def transform_xyz(X, R=2.0, r=1.0):
    """
    Convert from Cartesian coordinates (x, y, z) to torus angles (phi, theta).
    """
    xyz = np.asarray(X)
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    # phi 
    phi = np.arctan2(y, x)

    # theta = arctan2(z, sqrt(x^2 + y^2) - R)
    denom = np.sqrt(x**2 + y**2) - R
    theta = np.arctan2(z, denom)
    return phi, theta

def gelu(x):
    return F.gelu(x)

def ex_func(inputs):
    """Extra function in PDE: Source term f.
        Input: inputs, array of (x,y,z), shape: (N,3)
        Output: f. shape: (N,3). N is the number of points
    """
    # source term in the pde
    p, t = transform_xyz(inputs)
    p,t = np.array(p).reshape((-1,1)), np.array(t).reshape((-1,1))
    # print("p,t",p,t)
    gsin = np.sin(t)
    gcos = 2 + np.cos(t)
    c = 1.1 + np.sin(t)**2 * np.cos(p)**2
    c_t = 2*np.sin(t)*np.cos(t)*np.cos(p)**2
    c_p = - 2*np.sin(t)**2 *np.cos(p)*np.sin(p)
    u = np.sin(p) * np.sin(t)
    u_t = np.sin(p) * np.cos(t)
    u_tt = np.sin(p) * (- np.sin(t))
    u_p = np.cos(p) * np.sin(t)
    u_pp = - np.sin(p) * np.sin(t)
    f = -((-gsin*c*u_t + gcos*c_t*u_t + gcos*c*u_tt) + c_p/gcos*u_p + c/gcos*u_pp )/gcos+u

    return f.astype(np.float32)

def main(Numk, fn):
    def pde(x, y, sparse_mat, f):
        # Just a placeholder: return a zero residual
        return torch.zeros_like(y)

    # 1) Load entire dataset
    X_train, y_train, X_test, y_test = load_data()

    # 2) PDE residual data anchors
    N_k, N_p = 10, 2500
    anchors_train = load_train_data(f"{DNAME}/semitorus_data_train1_rbf.npz", N_k, N_p)[0]

    # 3) PDE test anchors
    N_kt, N_pt = 10, 2500
    anchors_test_data = load_test_data(f"{DNAME}/semitorus_data_test1_rbf.npz", N_kt, N_pt)

    # 4) Observation (BC) data
    N_ob_k, N_ob_p = Numk, 2500
    anchors_train_ob, anchors_train_ob_y = load_train_data(
        f"{DNAME}/semitorus_data_train1_rbf.npz", N_ob_k, N_ob_p
    )

    geom = dde.geometry.PointCloud(X_train[1]) 
    observe_y = dde.icbc.PointSetBC(anchors_train_ob[1], anchors_train_ob_y, component=0, shuffle=False)
    pde = dde.data.PDE(geom, pde, [observe_y], num_domain=0, auxiliary_var_function=ex_func)

    func_space = KappaFuncSpace() # kappa should be in the same form as oberservations
    x = np.linspace(-3, 3, num=26) # domain is -3,3 
    y = np.linspace(-3, 3, num=26) # Nkx = 26*26
    xv, yv = np.meshgrid(x, y)
    eval_pts = np.vstack((np.ravel(xv.T), np.ravel(yv.T))).T # for branch net
    
    print("Running PDEOperatorRBFCartesianProd...")
    data = PDEOperatorRBFCartesianProd(
        pde,
        func_space,
        eval_pts,
        num_function=0,
        anchors=anchors_train,
        anchors_test_data=anchors_test_data,
        function_variables=[0, 1],
        batch_size=None,
    )

    m = 26**2
    n = 32

    branch_net = nn.Sequential(
        nn.Unflatten(dim=1, unflattened_size=(1, 26, 26)),
        nn.Conv2d(1, 8, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Conv2d(8, 16, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, n),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(n, n),
    )

    net = dde.nn.DeepONetCartesianProd(
        [m, branch_net],
        [3, n, n, n], 
        {"branch": "relu", "trunk": gelu},
        "Glorot normal",
        multi_output_strategy=None,
    )
    
    
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, decay=("inverse time", 1000, 0.05),
        metrics=["mean l2 relative error"], loss_weights=[1e-2, 1.0], )
    t1 = time.time()
    losshistory, train_state = model.train(iterations=100000, batch_size=None)#, model_save_path="model/model.ckpt")
    t2 = time.time()
    print("Time of training:", t2 - t1)
    
    # Save the loss history
    dde.utils.save_loss_history(losshistory, f"rbf_res/loss_rbf_pi_100_{Numk}_{fn}.dat")
    dde.utils.plot_loss_history(losshistory, f"rbf_res/loss_rbf_pi_100_{Numk}_{fn}.png")

    pred_u = model.predict(X_test)
    mse_test = dde.metrics.mean_squared_error(
        y_test.reshape((-1, 1)), pred_u.reshape((-1, 1))
    )
    rel_test = dde.metrics.mean_l2_relative_error(y_test, pred_u)
    print("Test MSE =", mse_test)
    print("Test L2-rel =", rel_test)

    np.savetxt(f"rbf_res/pred_u_rbf_100_{Numk}_{fn}.txt", pred_u, delimiter=",")


if __name__ == "__main__":
    Numk = 2
    main(Numk, "0")

