import numpy as np
import time
import scipy
import scipy.sparse as sp

import torch
import torch.nn.functional as F

import deepxde as dde
import deepxde.backend as bkd
from deepxde.data import Data, function_spaces
from deepxde.utils import run_if_all_none
from sklearn.neighbors import NearestNeighbors

DNAME = "../../data/data_linear/"
torch_dtype = torch.float32
np_dtype = np.float32

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

def scipy_csr_to_torch_sparse(csr_mat):
    """
    Convert a SciPy CSR/COO matrix to a coalesced PyTorch sparse_coo_tensor.
    """
    coo = csr_mat.tocoo()
    indices = np.stack([coo.row, coo.col], axis=0)  # shape=(2, nnz)
    values = coo.data
    shape = coo.shape
    i_torch = torch.LongTensor(indices)
    v_torch = torch.tensor(values, dtype=torch_dtype)
    sparse_tensor = torch.sparse_coo_tensor(i_torch, v_torch, size=shape, device="cpu")
    return sparse_tensor.coalesce()

class PDEOperatorDMCartesianProd(Data):
    """
    Custom PDE operator data for the diffusion-map-based PDE
    (-div_g(kappa grad_g(u)) + u = f) in a point cloud setting.
    """

    def __init__(
        self,
        pde,
        function_space,
        evaluation_points,
        num_function,
        anchors=None,
        anchors_test_data=None,
        function_variables=None,
        num_test=None,
        batch_size=None,
    ):
        super().__init__()
        self.pde = pde
        self.func_space = function_space
        self.eval_pts = evaluation_points
        self.num_func = num_function
        self.func_vars = (
            function_variables
            if function_variables is not None
            else list(range(pde.geom.dim))
        )
        self.num_test = num_test
        self.batch_size = batch_size
        self.anchors = anchors
        self.anchors_test = anchors_test_data[0] if anchors_test_data else None
        self.anchors_test_y = anchors_test_data[1] if anchors_test_data else None

        self.dm_train = None
        self.dm_test = None

        self.train_x = None
        self.train_y = None
        self.train_aux_vars = None
        self.test_x = None
        self.test_y = None
        self.test_aux_vars = None

        # PDE geometry, BC, etc.
        self.bc = self.pde.bcs[0] 
        self.N = self.pde.train_x.shape[0]  # number of domain points
        self.eye_sparse = self._build_eye_sparse(self.N)

        self.train_next_batch()
        self.test()

        self.M = self.train_aux_vars.shape[0]
        self._build_pde_operators_train()
        self.f_const = self._build_rhs_constant(self.pde.train_x)

    def _build_eye_sparse(self, N):
        indices = torch.arange(N, dtype=torch.int32)
        indices_2d = torch.stack([indices, indices], dim=0)
        values = torch.ones(N, dtype=torch_dtype)
        shape = (N, N)
        return torch.sparse_coo_tensor(indices_2d, values, size=shape, device="cpu").coalesce()

    def _build_rhs_constant(self, x, training=True):
        ex_f = self.pde.train_aux_vars if training else self.pde.test_aux_vars
        return torch.tensor(ex_f, dtype=torch_dtype).to("cuda:0")

    def _build_pde_operators_train(self):
        self.operator_train = []
        for i in range(self.M):
            kappa_i = self.train_aux_vars[i, :]  # shape (N,)
            L_csr = self.dm_train.get_matrix(kappa_i)
            L_torch = scipy_csr_to_torch_sparse(L_csr)  # Coalesced sparse tensor
            L_torch = L_torch.to(L_torch.device) 
            device = L_torch.device
            #print("the device of L_torch ", device)
    
            # Convert L to => -kappa_i(row)*value
            row_indices = L_torch.indices()[0, :]
            kappa_tensor = torch.tensor(kappa_i, dtype=torch_dtype, device=device)
            row_indices = row_indices.to(device)
            kappa_values = torch.gather(kappa_tensor, 0, row_indices)
            val = L_torch.values() * -kappa_values
    
            L_mod = torch.sparse_coo_tensor(
                L_torch.indices(), val, size=L_torch.shape, device=device
            ).coalesce()
    
            # Add identity (ensure eye_sparse is on same device)
            eye_sparse = self.eye_sparse.to(device)
            op = self._sparse_add(L_mod, eye_sparse).to("cuda:0")
    
            self.operator_train.append(op)


    def _sparse_add(self, spA, spB):
        return (spA + spB).coalesce()

    def _build_pde_operators_test(self):
        if self.test_aux_vars is None or self.dm_test is None:
            return
        self.operator_test = []
        M_test = self.test_aux_vars.shape[0]
        for i in range(M_test):
            kappa_i = self.test_aux_vars[i, :]
            L_csr = self.dm_test.get_matrix(kappa_i)
            L_torch = scipy_csr_to_torch_sparse(L_csr)  # Now coalesced

            row_indices = L_torch.indices()[0, :]
            kappa_tensor = torch.tensor(kappa_i, dtype=torch_dtype)
            kappa_values = torch.gather(kappa_tensor, 0, row_indices)
            val = L_torch.values() * -kappa_values
            L_mod = torch.sparse_coo_tensor(
                L_torch.indices(), val, size=L_torch.shape, device="cpu"
            ).coalesce()

            op = self._sparse_add(L_mod, self.eye_sparse)
            self.operator_test.append(op)

    def _losses(self, outputs, loss_fn, inputs, model, num_func, training):
        pde_losses = []
        if training:
            operators = self.operator_train
        else:
            operators = self.operator_test
            if operators is None:
                self._build_pde_operators_test()
                operators = self.operator_test
        
        for i in range(num_func):
            u_i = outputs[i].to(outputs.device).unsqueeze(-1)
            r_i = torch.sparse.mm(operators[i], u_i) - self.f_const
            pde_losses.append(loss_fn(torch.zeros_like(r_i), r_i))
        pde_loss = bkd.reduce_mean(bkd.stack(pde_losses, 0))
        
        num_func_bc = self.bc.values.shape[0]
        bc_values = torch.tensor(self.bc.values, dtype=torch_dtype, device=outputs.device)
        bc_loss = loss_fn(bc_values, outputs[:num_func_bc])
    
        return [pde_loss, bc_loss]

    def losses_train(self, targets, outputs, loss_fn, inputs, model, aux=None):
        if self.anchors is not None:
            num_func = self.num_func + len(self.anchors[0])
        else:
            num_func = self.num_func
        return self._losses(outputs, loss_fn, inputs, model, num_func, training=True)

    def losses_test(self, targets, outputs, loss_fn, inputs, model, aux=None):
        if self.anchors_test is None or self.anchors_test_y is None:
            return [torch.tensor(0.0), torch.tensor(0.0)]
        loss = loss_fn(torch.tensor(self.anchors_test_y, dtype=torch_dtype, device=outputs.device), outputs)
        return [0.0*loss, loss]

    def train_next_batch(self, batch_size=None):
        #print("train_next_batch...")
        if self.train_x is None:
            func_feats = self.func_space.random(self.num_func)
            func_vals = self.func_space.eval_batch(func_feats, self.eval_pts)
            vx = self.func_space.eval_batch(
                func_feats, self.pde.train_x[:, self.func_vars]
            )

            if self.anchors is not None:
                anchor_kappa_vals = self.anchors[0]
                func_feats = self.func_space.random(self.num_func + len(anchor_kappa_vals))
                func_vals = np.vstack((anchor_kappa_vals, func_vals))

                vx_ob = np.loadtxt(f"{DNAME}kappast_train.txt", delimiter=",")
                vx_ob = vx_ob[: len(anchor_kappa_vals), :]
                vx = np.vstack((vx_ob, vx))

            self.train_x = (func_vals.astype(np_dtype), self.pde.train_x)
            self.train_aux_vars = vx.astype(np_dtype)

            if self.dm_train is None:
                self.dm_train = DiffusionMap(self.train_x[1], self.train_aux_vars)

        return self.train_x, self.train_y, self.train_aux_vars

    @run_if_all_none("test_x", "test_y", "test_aux_vars")
    def test(self):
        if self.num_test is None and self.anchors_test is None:
            self.test_x = self.train_x
            self.test_aux_vars = self.train_aux_vars
            self.dm_test = self.dm_train
        else:
            if self.num_test is None:
                self.num_test = 0
            func_feats = self.func_space.random(self.num_test)
            func_vals = self.func_space.eval_batch(func_feats, self.eval_pts)
            vx = self.func_space.eval_batch(
                func_feats, self.pde.test_x[:, self.func_vars]
            )
            # if anchor test data also present
            if self.anchors_test is not None:
                anchor_kappa_vals = self.anchors_test[0]  # shape (#anchors_test, #branch_points)
                func_feats = self.func_space.random(self.num_test + len(anchor_kappa_vals))
                func_vals = np.vstack((anchor_kappa_vals, func_vals))

                vx_ob = np.loadtxt(f"{DNAME}kappast_test.txt", delimiter=",")
                vx_ob = vx_ob[: len(anchor_kappa_vals), :]
                vx = np.vstack((vx_ob, vx))

            self.test_x = (func_vals.astype(np_dtype), self.pde.test_x)
            self.test_aux_vars = vx.astype(np_dtype)
            self.test_y = self.anchors_test_y

            if self.dm_test is None:
                self.dm_test = DiffusionMap(self.test_x[1], self.test_aux_vars)

        return self.test_x, self.test_y, self.test_aux_vars

class DiffusionMap:
    
    def __init__(self, X, kappa):
        self.X = X
        self.N = len(X)
        self.k = int(np.floor(1.5*np.sqrt(self.N)))
        distances, indices = self.knn(X)
        self.d = distances
        self.inds = indices
        self.epsilon = self.estimate_epsilon()
        print("epsilon:", self.epsilon)
        self.W, self.q = self.compute_Wq()

    def compute_Wq(self):
        """
        Base W ignoring kappa, row-symmetric kernel matrix in CSR.
        """
        W0 = np.exp(-self.d**2 / (4 * self.epsilon))  # shape (N,k)
        row_inds = np.repeat(np.arange(self.N), self.k)
        col_inds = self.inds.flatten()
        vals = W0.flatten()

        W = sp.csr_matrix((vals, (row_inds, col_inds)), shape=(self.N, self.N))
        W = 0.5 * (W + W.transpose())
        q = np.array(W.sum(axis=0)).ravel()
        return W, q

    def get_matrix(self, kappa_i):
        """
        Build final operator L = (D2^-1 W_tilde) - I) / epsilon.
        Weighted by -diag(kappa_i).
        """
        W, q = self.W, self.q
        # scale columns by kappa_i / q
        scale_cols = kappa_i / q
        Sc = sp.diags(scale_cols, format="csr")
        W_scaled = W @ Sc  # each column j scaled by kappa_i[j]/q[j]

        # left normalization by sqrt of row sums
        D1 = np.array(W_scaled.sum(axis=1)).ravel()
        invsqrt_D1 = 1.0 / np.sqrt(D1 + 1e-14)
        Ds = sp.diags(invsqrt_D1, format="csr")
        W_tilde = W_scaled @ Ds

        D2 = np.array(W_tilde.sum(axis=1)).ravel()
        invD2 = sp.diags(1.0 / (D2 + 1e-14), format="csr")
        L = invD2 @ W_tilde
        I = sp.eye(self.N, format="csr")
        L = L - I
        L = L / self.epsilon
        return L.astype(np_dtype).tocsr()

    def knn(self, X):
        nbrs = NearestNeighbors(n_neighbors=self.k).fit(X)
        distances, indices = nbrs.kneighbors(X)
        return distances, indices

    def estimate_epsilon(self):
        eps_candidates = [2 ** (i / 10) for i in range(-300, 101)]
        dpreGlobal = []
        for eps in eps_candidates:
            val = np.sum(np.exp(-self.d**2 / (4 * eps))) / (self.N * self.k)
            dpreGlobal.append(val)

        dpreGlobal = np.array(dpreGlobal).reshape(1, -1)
        log_dg = np.log(dpreGlobal)
        log_eps = np.log(eps_candidates).reshape(1, -1)

        halfdim = np.diff(log_dg) / np.diff(log_eps)
        max_val = np.max(halfdim)
        max_ind = np.argmax(halfdim)
        epsilon = eps_candidates[max_ind]
        return epsilon

class KappaFuncSpace(function_spaces.FunctionSpace):
    """
    Example custom kappa function space:
    kappa = a1*x^2 + b1*y^2 + a2*x + b2*y + 3 + c
    Each coefficient is sampled randomly from [0, M].
    """

    def __init__(self, M=1, type="Nonlinear"):
        super().__init__()
        self.type = type
        self.M = M

    def random(self, size):
        """
        Return shape (size, 5) for [a1, b1, a2, b2, c].
        """
        return np.random.rand(size, 5) * self.M

    def eval_one(self, feature, x):
        a1, b1, a2, b2, c = feature
        X = x[:, 0]
        Y = x[:, 1]
        return a1 * X**2 + b1 * Y**2 + a2 * X + b2 * Y + 3.0 + c

    def eval_batch(self, features, xs):
        """
        features shape: (num_funcs, 5)
        xs shape: (num_points, 2)
        Return kappa values shape: (num_funcs, num_points).
        """
        a1 = features[:, 0].reshape(-1, 1)
        b1 = features[:, 1].reshape(-1, 1)
        a2 = features[:, 2].reshape(-1, 1)
        b2 = features[:, 3].reshape(-1, 1)
        c = features[:, 4].reshape(-1, 1)
        X = xs[:, 0].reshape(1, -1)
        Y = xs[:, 1].reshape(1, -1)

        Ks = a1 * X**2 + b1 * Y**2 + a2 * X + b2 * Y + 3.0 + c
        return Ks
