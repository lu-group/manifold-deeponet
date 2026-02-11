import numpy as np
import torch
import matplotlib.pyplot as plt

from dm_utils_pytorch import DiffusionMap, scipy_csr_to_torch_sparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dname = "../../data/data_bunny_8k"
dtype = torch.float64

def convert_sparse_matrix_to_sparse_tensor(X, device=device):
    """
    Convert a SciPy CSR/COO matrix to a coalesced PyTorch sparse_coo_tensor.
    """
    coo = X.tocoo()
    indices = np.vstack((coo.row, coo.col))
    values = coo.data
    shape = coo.shape
    
    i_torch = torch.LongTensor(indices).to(device)
    v_torch = torch.tensor(values, dtype=dtype, device=device)
    sparse_tensor = torch.sparse_coo_tensor(i_torch, v_torch, size=shape, device=device)
    return sparse_tensor.coalesce()

def compute_res(u, ind_k):
    X = np.loadtxt(f"{dname}/Xt_8k.txt", delimiter=",")
    kappa_full = np.loadtxt(f"{dname}/kappast_train.txt", delimiter=",")
    dm = DiffusionMap(X, kappa_full)
    u = torch.tensor(u, dtype=dtype, device=device)
    N = X.shape[0]
    M = kappa_full.shape[0]
    # _build_eye_sparse
    indices = torch.arange(N, dtype=torch.int64, device=device)
    indices_2d = torch.stack([indices, indices], dim=0)
    values = torch.ones(N, dtype=dtype, device=device)
    shape = (N, N)
    eye_sparse = torch.sparse_coo_tensor(indices_2d, values, size=shape, device=device).coalesce()
    # _build_pde_operators_train
    operator_train = []
    for i in range(M):
        kappa_i = kappa_full[i, :]  # shape (N,)
        L_csr = dm.get_matrix(kappa_i)
        L_torch = scipy_csr_to_torch_sparse(L_csr) 
        row_indices = L_torch.indices()[0, :]
        kappa_tensor = torch.tensor(kappa_i, dtype=dtype, device=device)
        kappa_values = torch.gather(kappa_tensor, 0, row_indices)
        val = L_torch.values() * -kappa_values
        L_mod = torch.sparse_coo_tensor(
            L_torch.indices(), val, size=L_torch.shape, device=device
        ).coalesce()
        op = (L_mod + eye_sparse).coalesce()
        operator_train.append(op)
    # _build_rhs_constant
    sum_xyz = np.sum(X[:, :3], axis=1, keepdims=True)
    f_const =  torch.tensor(0.1 * (sum_xyz**2), dtype=dtype, device=device)

    r_ind_k = torch.sparse.mm(operator_train[ind_k], u[ind_k].unsqueeze(-1))- f_const 
    res = r_ind_k.cpu().detach().numpy()
    mse = np.mean((np.zeros((N, 1)) - res.reshape((N, 1)))**2)
    print("Residual MSE:", mse)

    # Visualization
    TRI = np.loadtxt(f"{dname}/TRI_8k.txt", delimiter=",", dtype=int)
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    face_colors = np.mean(res.flatten()[(TRI - 1)], axis=1)

    trisurf = ax.plot_trisurf(
        X[:, 0], X[:, 2], X[:, 1],
        triangles=TRI - 1,
        cmap='rainbow',
        linewidth=0.2,
        antialiased=True
    )
    trisurf.set_array(face_colors)
    trisurf.set_clim(face_colors.min(), face_colors.max())
    trisurf.autoscale()
    ax.set_xlabel('$x$', labelpad=15) 
    ax.set_ylabel('$z$', labelpad=15) 
    ax.set_zlabel('$y$', labelpad=15) 
    ax.view_init(elev=20, azim=100) 
    fig.colorbar(trisurf, ax=ax, shrink=0.5, aspect=10)
    plt.savefig(f"residual_kappa_{ind_k}.png", dpi=300)
    plt.show()

# Load input u and compute residual
u = np.loadtxt(f"{dname}/us_train.txt", delimiter=",")
compute_res(u, 10)
