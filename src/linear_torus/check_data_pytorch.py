import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from dm_utils_pi import DiffusionMap, scipy_csr_to_torch_sparse
import scipy.interpolate

DNAME = "../../data/data_linear"  # e.g., "data_higher_order", "data_exp", etc.
DEVICE = "cpu"
FLOAT_DTYPE = torch.float64

fontsize=25
fig_params = {
    'font.size': fontsize,
    "savefig.dpi": 300, 
    "figure.figsize": (8, 6),
    'lines.linewidth': 2.5,
    'axes.linewidth': 2.5,
    'axes.titlesize' : fontsize+5,
    'xtick.direction':'in',
    'ytick.direction':'in',
    'xtick.major.size': 7,
    'xtick.minor.size': 5,
    'xtick.major.width': 3,
    'xtick.minor.width': 2,
    'xtick.major.pad': 6,
    'xtick.minor.pad': 5,
    'ytick.major.pad': 6,
    'ytick.minor.pad': 5,
    'ytick.major.size': 7,
    'ytick.minor.size': 5,
    'ytick.major.width': 3,
    'ytick.minor.width': 2,
    'legend.frameon' : False,
    'ytick.left': True,
    'ytick.labelleft': True,
    'xtick.bottom': True,
    'xtick.labelbottom': True
}
plt.rcParams.update(fig_params)

def convert_sparse_matrix_to_sparse_tensor(sparse_mat, device=DEVICE):
    """
    Convert a SciPy CSR/COO matrix to a coalesced PyTorch sparse_coo_tensor.
    """
    coo = sparse_mat.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    shape = coo.shape

    i_torch = torch.LongTensor(indices).to(device)
    v_torch = torch.tensor(values, dtype=FLOAT_DTYPE, device=device)
    sparse_tensor = torch.sparse_coo_tensor(i_torch, v_torch, size=shape, device=device)
    return sparse_tensor.coalesce()

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

def compute_res(u, ind_k):
    """
    Compute the residual of the PDE for a specific kappa index.

    Args:
        u (np.ndarray): Shape (#functions, #points), the solution array
        ind_k (int): Index of the kappa function

    Returns:
        np.ndarray: Residual values
    """
    Xt_train_path = os.path.join(DNAME, "Xt_train.txt")
    kappast_train_path = os.path.join(DNAME, "kappast_train.txt")
    thetat_path = os.path.join(DNAME, "thetat.txt")
    phit_path = os.path.join(DNAME, "phit.txt")

    X = np.loadtxt(Xt_train_path, delimiter=",")
    kappa_full = np.loadtxt(kappast_train_path, delimiter=",")
    dm = DiffusionMap(X, kappa_full)

    kappa_selected = kappa_full[ind_k, :].astype(np.float32)
    u_selected = torch.tensor(u[ind_k, :], dtype=FLOAT_DTYPE, device=DEVICE).unsqueeze(-1)

    L_csr = dm.get_matrix(kappa_selected)
    L_torch = scipy_csr_to_torch_sparse(L_csr).to(DEVICE).type(FLOAT_DTYPE)

    N = L_csr.shape[0]
    eye_indices = torch.arange(N, dtype=torch.long, device=DEVICE)
    eye_indices = torch.stack([eye_indices, eye_indices], dim=0)
    eye_values = torch.ones(N, dtype=FLOAT_DTYPE, device=DEVICE)
    eye_sparse = torch.sparse_coo_tensor(eye_indices, eye_values, size=(N, N), device=DEVICE).coalesce()

    L_indices = L_torch.indices()
    L_values = L_torch.values() * (-kappa_selected[L_indices[0, :]])
    L_scaled = torch.sparse_coo_tensor(L_indices, L_values, size=L_torch.size(), device=DEVICE).coalesce()

    # Operator: (-diag(kappa)*L) + I
    operator = L_scaled + eye_sparse

    f_vals = ex_func(X).reshape(-1, 1)
    f_tensor = torch.tensor(f_vals, dtype=FLOAT_DTYPE, device=DEVICE)

    res_tensor = torch.sparse.mm(operator, u_selected) - f_tensor

    res_numpy = res_tensor.cpu().detach().numpy()
    mse = np.mean((0 - res_numpy)**2)
    print(f"Residual MSE for kappa {ind_k}: {mse}")

    # Visualization
    t = np.loadtxt(thetat_path, delimiter=",")
    p = np.loadtxt(phit_path, delimiter=",")
    pi, ti = np.linspace(p.min(), p.max(), 100), np.linspace(t.min(), t.max(), 100)
    pi, ti = np.meshgrid(pi, ti)


    rbf = scipy.interpolate.Rbf(p, t, res_numpy, function='cubic')
    zi = rbf(pi, ti)

    plt.figure()
    plt.imshow(zi, interpolation='gaussian',
               extent=[p.min(), p.max(), t.max(), t.min()],
               origin="lower")
    plt.colorbar()
    plt.xlabel("$\\varphi$")
    plt.ylabel("$\\theta$")
    plt.savefig(f"residual_kappa_{ind_k}.png", bbox_inches='tight')
    plt.show()

    plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_aspect('auto')
    ax.set_zlim(-3, 3)
    scatter = ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=res_numpy, cmap='viridis')
    fig = plt.gcf()
    fig.colorbar(scatter)
    plt.savefig(f"residual_3d_kappa_{ind_k}.png", bbox_inches='tight')
    plt.show()

def main():
    us_train_path = os.path.join(DNAME, "us_train.txt")
    u = np.loadtxt(us_train_path, delimiter=",")
    residual = compute_res(u, 1)


if __name__ == "__main__":
    main()
