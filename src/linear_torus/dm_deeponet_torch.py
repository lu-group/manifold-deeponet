import os
import time
import numpy as np
import deepxde as dde
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from dm_utils_ob import PDEOperatorDMCartesianProd, KappaFuncSpace
dde.config.disable_xla_jit()
dde.config.set_default_float("float32")

DNAME = "../../data/data_linear/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "dm_res"
iterations = 100000
loss_weights=[0, 1.0]
print(" loss_weights:", loss_weights)
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    d = np.load(f"{DNAME}/torus_data_train1_dm.npz", allow_pickle=True)
    X_train_1, X_train_2, y_train = d["X_train_branch"], d["X_train_trunk"], d["y_train"]
    X_train = (X_train_1, X_train_2)

    d = np.load(f"{DNAME}/torus_data_test1_dm.npz", allow_pickle=True)
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
    return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def ex_func(inputs):
    """Extra function in PDE: Source term f.
        Input: inputs, array of (x,y,z), shape: (N,3)
        Output: f. shape: (N,3). N is the number of points
    """
    # source term in the pde
    p, t = transform_xyz(inputs)
    p = p.reshape(-1, 1)
    t = t.reshape(-1, 1)

    gsin = np.sin(t)
    gcos = 2 + np.cos(t)
    c = 1.1 + np.sin(t)**2 * np.cos(p)**2
    c_t = 2 * np.sin(t) * np.cos(t) * np.cos(p)**2
    c_p = -2 * np.sin(t)**2 * np.cos(p) * np.sin(p)

    u = np.sin(p) * np.sin(t)
    u_t = np.sin(p) * np.cos(t)
    u_tt = np.sin(p) * (-np.sin(t))
    u_p = np.cos(p) * np.sin(t)
    u_pp = -np.sin(p) * np.sin(t)

    f = ((-gsin * c * u_t + gcos * c_t * u_t + gcos * c * u_tt) +
         c_p / gcos * u_p + c / gcos * u_pp) / gcos - u

    return f.astype(np.float32)

def main(Numk, n, convs, D, fn):
    def pde(x, y, sparse_mat, f):
        # Just a placeholder
        return torch.zeros_like(y)

    # 1) Load entire dataset
    X_train, y_train, X_test, y_test = load_data()

    # 2) PDE residual data anchors
    N_k, N_p = 100, 2500
    anchors_train = load_train_data(f"{DNAME}/torus_data_train1_dm.npz", N_k, N_p)[0]

    # 3) PDE test anchors
    N_kt, N_pt = 100, 2500
    anchors_test_data = load_test_data(f"{DNAME}/torus_data_test1_dm.npz", N_kt, N_pt)

    # 4) Observation (BC) data
    N_ob_k, N_ob_p = Numk, 2500
    anchors_train_ob, anchors_train_ob_y = load_train_data(
        f"{DNAME}/torus_data_train1_dm.npz", N_ob_k, N_ob_p
    )

    geom = dde.geometry.PointCloud(X_train[1]) 
    observe_y = dde.icbc.PointSetBC(anchors_train_ob[1], anchors_train_ob_y, component=0, shuffle=False)
    pde = dde.data.PDE(geom, pde, [observe_y], num_domain=0, auxiliary_var_function=ex_func)

    func_space = KappaFuncSpace() # kappa should be in the same form as oberservations
    x = np.linspace(-3, 3, num=26) # domain is -3,3 
    y = np.linspace(-3, 3, num=26) # Nkx = 26*26
    xv, yv = np.meshgrid(x, y)
    eval_pts = np.vstack((np.ravel(xv.T), np.ravel(yv.T))).T # for branch net
    
    data = PDEOperatorDMCartesianProd(
        pde, func_space, eval_pts, num_function=0,
        anchors=anchors_train,
        anchors_test_data=anchors_test_data,
        function_variables=[0, 1], batch_size=None)

    m = 26**2
    
    def build_branch_net(n, convs=[16, 32]):
        H = W = 26
        in_chs = 1
        layers = [nn.Unflatten(dim=1, unflattened_size=(in_chs, H, W))]
        
        for out_chs in convs:
            print("conv layer", in_chs, out_chs)
            layers += [
                nn.Conv2d(in_chs, out_chs, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Dropout(p=0.3),
            ]
            H = (H - 3) // 2 + 1
            W = (W - 3) // 2 + 1
            in_chs = out_chs
        
        layers += [
            nn.Flatten(),
            nn.Linear(in_chs * H * W, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, n),
        ]
        return nn.Sequential(*layers)
    
    branch_net = build_branch_net(n=n, convs=convs)
    
    net = dde.nn.DeepONetCartesianProd(
        [m, branch_net],
        [3] + [n]*D, 
        {"branch": "relu", "trunk": gelu},
        "Glorot normal",
        multi_output_strategy=None,
    )
    
    
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, decay=("inverse time", 1000, 0.05),
        metrics=["mean l2 relative error"], loss_weights=loss_weights, )
    print("# Parameters:", net.num_trainable_parameters())
    t1 = time.time()
    losshistory, train_state = model.train(iterations=iterations, batch_size=None)#, model_save_path="model/model.ckpt")
    t2 = time.time()
    print("Time of training:", t2 - t1)
    
    # Save the loss history
    dde.utils.save_loss_history(losshistory, f"{OUTPUT_DIR}/loss_dm_ob_0_{Numk}_{n}_{D}_{convs}_{fn}.dat")
    dde.utils.plot_loss_history(losshistory, f"{OUTPUT_DIR}/loss_dm_ob_0_{Numk}_{n}_{D}_{convs}_{fn}.png")

    pred_u = model.predict(X_test)
    mse_test = dde.metrics.mean_squared_error(
        y_test.reshape((-1, 1)), pred_u.reshape((-1, 1))
    )
    rel_test = dde.metrics.mean_l2_relative_error(y_test, pred_u)
    print("Test MSE =", mse_test)
    print("Test L2-rel =", rel_test)

    np.savetxt(f"{OUTPUT_DIR}/pred_u_dm_0_{Numk}_{n}_{D}_{convs}_{fn}.txt", pred_u, delimiter=",")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--numk",   type=int, default=25)
    parser.add_argument("--n",      type=int, default=32)
    parser.add_argument("--convs",  nargs="+", type=int, default=[16,32])
    parser.add_argument("--D",      type=int, default=3)
    parser.add_argument("--fn",      type=str, default="0")
    
    args = parser.parse_args()
    print("args:", args)
    main(args.numk, args.n, args.convs, args.D, args.fn)