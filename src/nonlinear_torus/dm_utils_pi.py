import numpy as np
import scipy
import torch
import deepxde.backend as bkd
from deepxde.data import Data, function_spaces
from deepxde.utils import run_if_all_none
from sklearn.neighbors import NearestNeighbors

dname = "../../data/data_nonlinear/"
dtype = torch.float32
npdtype = np.float32

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
    v_torch = torch.tensor(values, dtype=dtype)
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

        self.bc = self.pde.bcs[0] 
        self.N = self.pde.train_x.shape[0]  # number of domain points
        self.eye_sparse = self._build_eye_sparse(self.N)

        self.train_next_batch()
        self.test()

        self.M = self.train_aux_vars.shape[0]
        self._build_pde_operators_train()

    def _build_eye_sparse(self, N):
        indices = torch.arange(N, dtype=torch.int32)
        indices_2d = torch.stack([indices, indices], dim=0)
        values = torch.ones(N, dtype=dtype)
        shape = (N, N)
        return torch.sparse_coo_tensor(indices_2d, values, size=shape, device="cpu").coalesce()

    def _build_pde_operators_train(self):
        self.operator_train = []
        self.kappas = torch.tensor(self.train_aux_vars, dtype=dtype).to("cuda:0")
        for i in range(self.M):
            kappa_i = self.train_aux_vars[i, :]  # shape (N,)
            L_csr = self.dm_train.get_matrix(kappa_i)
            L_torch = scipy_csr_to_torch_sparse(L_csr)  # Coalesced sparse tensor
            L_torch = L_torch.to(L_torch.device) 
            device = L_torch.device
            #print("the device of L_torch ", device)
    
            # Convert L to => -kappa_i(row)*value
            row_indices = L_torch.indices()[0, :]
            kappa_tensor = torch.tensor(kappa_i, dtype=dtype, device=device)
            row_indices = row_indices.to(device)
            kappa_values = torch.gather(kappa_tensor, 0, row_indices)
            val = L_torch.values() 
    
            L_mod = torch.sparse_coo_tensor(
                L_torch.indices(), -val, size=L_torch.shape, device=device
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
            L_torch = scipy_csr_to_torch_sparse(L_csr)

            row_indices = L_torch.indices()[0, :]
            kappa_tensor = torch.tensor(kappa_i, dtype=dtype)
            kappa_values = torch.gather(kappa_tensor, 0, row_indices)
            val = L_torch.values() * -kappa_values
            L_mod = torch.sparse_coo_tensor(
                L_torch.indices(), val, size=L_torch.shape, device="cpu"
            ).coalesce()

            op = self._sparse_add(L_mod, self.eye_sparse)
            self.operator_test.append(op)

    def _losses(self, outputs, loss_fn, inputs, model, num_func, training):
        #print("outputs.device:", outputs.device)
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
            kappa_reshaped = self.kappas[i].unsqueeze(-1)
            f_const = 1.5 * (u_i ** 2) + 2 * kappa_reshaped * u_i - 0.5 * (kappa_reshaped ** 2) + u_i
            r_i = torch.sparse.mm(operators[i], u_i) - f_const
            pde_losses.append(loss_fn(torch.zeros_like(r_i), r_i))
        pde_loss = bkd.reduce_mean(bkd.stack(pde_losses, 0))
        
        num_func_bc = self.bc.values.shape[0]
        bc_values = torch.tensor(self.bc.values, dtype=dtype, device=outputs.device)
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
        loss = loss_fn(torch.tensor(self.anchors_test_y, dtype=dtype, device=outputs.device), outputs)
        return [0.0*loss, loss]

    def train_next_batch(self, batch_size=None):
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

                vx_ob = np.loadtxt(f"{dname}kappast_train.txt", delimiter=",")
                vx_ob = vx_ob[: len(anchor_kappa_vals), :]
                vx = np.vstack((vx_ob, vx))

            self.train_x = (func_vals.astype(npdtype), self.pde.train_x)
            self.train_aux_vars = vx.astype(npdtype)

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

                vx_ob = np.loadtxt(f"{dname}kappast_test.txt", delimiter=",")
                vx_ob = vx_ob[: len(anchor_kappa_vals), :]
                vx = np.vstack((vx_ob, vx))

            self.test_x = (func_vals.astype(npdtype), self.pde.test_x)
            self.test_aux_vars = vx.astype(npdtype)
            self.test_y = self.anchors_test_y

            if self.dm_test is None:
                self.dm_test = DiffusionMap(self.test_x[1], self.test_aux_vars)

        return self.test_x, self.test_y, self.test_aux_vars

class DiffusionMap:
    def __init__(self, X, kappa):
        """Calculate L matrix.
           Inputs:
            X: self.pde.train_x (N,3)
            kappa: kappa values at X (on torus)
        """
        self.N = len(X)
        distances, indices = self.knn(X)
        # self.kappa = np.float32(kappa)
        # self.Nk = len(kappa)
        self.d = distances
        self.inds = indices
        self.epsilon = self.estimate_epsilon(X)
        print("epsilon", self.epsilon)

    def get_matrix(self, kappa_i):
        distances = self.d
        N = self.N
        k = self.k
        c = kappa_i.reshape((-1,1))
        W0 = np.exp(-distances**2/(4*self.epsilon)) # (N,k)
        i = np.transpose(self.inds).reshape((N*k, ), order = "F")
        j = np.tile(range(0,N), (k, 1)).reshape((N*k, ), order = "F")
        v = np.transpose(W0).reshape((N*k, ), order = "F")
        W = np.transpose(self.to_sparse(i, j, v, N, N))
        q = np.sum(W, axis=1)
        K =np.multiply(W*np.diag(np.array(1/q).flatten()), np.sqrt(c)*np.sqrt(c.transpose()))
        DD = np.diag(np.sum(K,axis=1))
        L= (K-DD)/self.epsilon
        L = np.float32(L)
        L = scipy.sparse.csr_matrix(L)
        return L

    def knn(self,X):
        self.k = int(np.floor(1.5*np.sqrt(self.N)))
        nbrs = NearestNeighbors(n_neighbors=self.k).fit(X)
        distances, indices = nbrs.kneighbors(X) #distances.shape (N, k), indices.shape (N,k)
        return distances, indices

    def estimate_epsilon(self, X):
        epss = [2**(i/10) for i in range(-300, 101)]
        dpreGlobal = []
        for ll in range(len(epss)):
            dpreGlobali = np.sum(np.sum(np.exp(-self.d**2/(4*epss[ll])), axis = 0).reshape((1,-1)), axis = 1)/(self.N*self.k)
            dpreGlobal.append(dpreGlobali)
        dpreGlobal = np.array(dpreGlobal).reshape((1,-1)) # (1, 401)
        halfdim = np.diff(np.log(dpreGlobal))/np.diff(np.log(epss).reshape((1, -1))) # (1,400)
        maxval,maxind = max(halfdim), halfdim.argmax()
        epsilon = epss[maxind]
        return epsilon
    
    def to_sparse(self, i,j,v,m,n):
        return scipy.sparse.csr_matrix((v, (i, j)), shape=(m, n))

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
