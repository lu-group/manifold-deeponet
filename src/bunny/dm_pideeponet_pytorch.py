import os
os.makedirs("dm_res", exist_ok=True)
import time
import numpy as np
import deepxde as dde
dde.backend.set_default_backend("pytorch")

import torch
import torch.nn as nn
import torch.nn.functional as F

from dm_utils_pytorch import PDEOperatorDMCartesianProd, KappaFuncSpace

dde.config.disable_xla_jit()
dde.config.set_default_float("float64")


dname = "../../data/data_bunny_8k"

def load_test_data(fname, m, n):
    """
    Load data for testing.
    Inputs: fname - file name
            m - number of kappas
            n - number of collocation points
    """
    if m == 0 and n == 0:
        return None, None
    d = np.load(fname, allow_pickle=True)
    X_test_1, X_test_2, y_test = d["X_test_branch"], d["X_test_trunk"], d["y_test"]
    X_test = (X_test_1[:m, :], X_test_2[:n, :])
    return X_test, y_test[:m, :n]


def load_train_data(fname, m, n):
    """
    Load data for training.
    Inputs: fname - file name
            m - number of kappas
            n - number of collocation points
    """
    if m == 0 and n == 0:
        return None, None
    d = np.load(fname, allow_pickle=True)
    X_train_1, X_train_2, y_train = d["X_train_branch"], d["X_train_trunk"], d["y_train"]
    X_train = (X_train_1[:m, :], X_train_2[:n, :])
    return X_train, y_train[:m, :n]


def load_data():
    """
    Load and return the entire dataset for both training and testing.
    """
    d = np.load(f"{dname}/bunny_train_dm.npz", allow_pickle=True)
    X_train_1, X_train_2, y_train = d["X_train_branch"], d["X_train_trunk"], d["y_train"]
    X_train = (X_train_1, X_train_2)

    d = np.load(f"{dname}/bunny_test_dm.npz", allow_pickle=True)
    X_test_1, X_test_2, y_test = d["X_test_branch"], d["X_test_trunk"], d["y_test"]
    X_test = (X_test_1, X_test_2)
    return X_train, y_train, X_test, y_test


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def main(Numk, fn):
    # PDE definition: -div_g(kappa grad_g(u)) + u = f

    def pde_placeholder(x, y, sparse_mat, f):
        # Just a placeholder: return a zero residual
        return torch.zeros_like(y)

    # 1) Load entire dataset
    X_train, y_train, X_test, y_test = load_data()

    # 2) PDE residual data anchors
    N_k, N_p = 10, 7986
    anchors_train = load_train_data(f"{dname}/bunny_train_dm.npz", N_k, N_p)[0]

    # 3) PDE test anchors
    N_kt, N_pt = 10, 7986
    anchors_test_data = load_test_data(f"{dname}/bunny_test_dm.npz", N_kt, N_pt)

    # 4) Observation data
    N_ob_k, N_ob_p = Numk, 7986
    anchors_train_ob, anchors_train_ob_y = load_train_data(
        f"{dname}/bunny_train_dm.npz", N_ob_k, N_ob_p
    )

    geom = dde.geometry.PointCloud(X_train[1])
    observe_y = dde.icbc.PointSetBC(
        anchors_train_ob[1], anchors_train_ob_y, component=0, shuffle=False
    )

    pde_data = dde.data.PDE(
        geom,
        pde_placeholder,
        bcs=[observe_y],
        num_domain=0,
        train_distribution="pseudo",
    )

    func_space = KappaFuncSpace()
    x_grid = np.linspace(-2, 1.5, num=26)
    y_grid = np.linspace(0.5, 4, num=26)
    xv, yv = np.meshgrid(x_grid, y_grid)
    eval_pts = np.vstack((xv.ravel(), yv.ravel())).T  # shape (26*26, 2)

    data = PDEOperatorDMCartesianProd(
        pde_data,
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
        nn.Conv2d(8, 16, kernel_size=3, stride=2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * 5 * 5, n),
        nn.ReLU(),
        nn.Linear(n, n),
    )

    net = dde.nn.DeepONetCartesianProd(
        [m, branch_net],
        [3, n, n], 
        {"branch": "relu", "trunk": gelu},
        "Glorot normal",
        multi_output_strategy=None,
    )
    
    model = dde.Model(data, net)
    model.compile(
        "adam",
        lr=0.001,
        decay=("inverse time", 20000, 0.1),
        loss_weights=[1e-5, 1.0],   # PDE residual vs. boundary/obs
    )

    t1 = time.time()
    losshistory, train_state = model.train(
        iterations=100000,
        batch_size=None,
    )
    t2 = time.time()
    print("Time of training:", t2 - t1)

    dde.utils.save_loss_history(losshistory, f"dm_res/loss_pi_10_{Numk}_{fn}.dat")
    dde.utils.plot_loss_history(losshistory, f"dm_res/loss_pi_10_{Numk}_{fn}.png")

    pred_u = model.predict(X_test)
    mse_test = dde.metrics.mean_squared_error(
        y_test.reshape((-1, 1)), pred_u.reshape((-1, 1))
    )
    rel_test = dde.metrics.mean_l2_relative_error(y_test, pred_u)
    print("Test MSE =", mse_test)
    print("Test L2-rel =", rel_test)
    np.savetxt(f"dm_res/dm_pred_u_pi_10_{Numk}_{fn}.txt", pred_u, delimiter=",")


if __name__ == "__main__":
    Numk = 2
    main(Numk, "1")
