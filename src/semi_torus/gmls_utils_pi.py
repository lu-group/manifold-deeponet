import numpy as np
import scipy
import deepxde.backend as bkd
from deepxde.data import Data, function_spaces
from deepxde.backend import torch
from deepxde.utils import run_if_all_none, array_ops_compat

dname = "../../data/data_semi/"

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

class PDEOperatorRBFCartesianProd(Data):
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
        self.pde = pde
        self.func_space = function_space
        self.eval_pts = evaluation_points
        self.num_func = num_function
        self.func_vars = (function_variables if function_variables is not None 
                          else list(range(pde.geom.dim)))
        self.num_test = num_test
        self.batch_size = batch_size
        self.anchors = anchors
        if anchors_test_data:
            self.anchors_test = anchors_test_data[0]
            self.anchors_test_y = anchors_test_data[1]
        else:
            self.anchors_test = None
            self.anchors_test_y = None

        self.train_x = None
        self.train_y = None
        self.train_aux_vars = None
        self.test_x = None
        self.test_y = None
        self.test_aux_vars = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bc = self.pde.bcs[0]
        self.N = self.pde.train_x.shape[0]

        self.operator_train = []
        self.operator_test = []

        self.Gx, self.Gy, self.Gz,  self.Lap, self.ind_i = self.get_G_matrix()
        self.I_N = torch.eye(self.N, dtype=torch.float32, device=self.device)

        self.ind_i = torch.tensor(self.ind_i, dtype=torch.long, device=self.device) - 1
        self.all_idx = torch.arange(self.N, device=self.device)
        self.mask = torch.ones(self.N, dtype=torch.bool, device=self.device)
        self.mask[self.ind_i] = False

        self.train_next_batch()
        self.test()

        self.M = self.train_aux_vars.shape[0]
        self._build_pde_operators_train()
        self.f_const = self._build_rhs_constant(self.pde.train_x)

    
    def get_G_matrix(self):
        Gx_ = np.loadtxt(f"{dname}/Gx.txt", delimiter=",").astype(np.float32)
        Gy_ = np.loadtxt(f"{dname}/Gy.txt", delimiter=",").astype(np.float32)
        Gz_ = np.loadtxt(f"{dname}/Gz.txt", delimiter=",").astype(np.float32)
        Lap_ = np.loadtxt(f"{dname}/Lap.txt", delimiter=",").astype(np.float32)
        ind_i = np.loadtxt(f"{dname}/index_I.txt", delimiter=",")  # shape e.g. (2433,)

        Gx_t = torch.tensor(Gx_, dtype=torch.float32)
        Gy_t = torch.tensor(Gy_, dtype=torch.float32)
        Gz_t = torch.tensor(Gz_, dtype=torch.float32)
        Lap_t = torch.tensor(Lap_, dtype=torch.float32)

        return Gx_t, Gy_t, Gz_t, Lap_t, ind_i

    def _build_pde_operators_train(self):
        self.operator_train = []
        N = self.N
        for i in range(self.M):
            kappa_i = torch.tensor(self.train_aux_vars[i], dtype=torch.float32, device=self.device).unsqueeze(1)
            ggx = (self.Gx.mm(kappa_i)) * self.Gx
            ggy = (self.Gy.mm(kappa_i)) * self.Gy
            ggz = (self.Gz.mm(kappa_i)) * self.Gz
            L_mat = - (ggx + ggy + ggz) - kappa_i * self.Lap + self.I_N
            L_in0 = L_mat.index_select(dim=0, index=self.ind_i)
            L_in = L_in0.index_select(dim=1, index=self.ind_i)
            self.operator_train.append(L_in)

    def _build_pde_operators_test(self):
        self.operator_test = []
        if self.test_aux_vars is None:
            return
        N = self.N
        for i in range(self.test_aux_vars.shape[0]):
            kappa_i = torch.tensor(self.test_aux_vars[i], dtype=torch.float32, device=self.device).unsqueeze(1)
            ggx = (self.Gx.mm(kappa_i)) * self.Gx
            ggy = (self.Gy.mm(kappa_i)) * self.Gy
            ggz = (self.Gz.mm(kappa_i)) * self.Gz
            gg_sum = ggx + ggy + ggz
            L_mat = - gg_sum - kappa_i * self.Lap + self.I_N
            L_in0 = L_mat.index_select(dim=0, index=self.ind_i)
            L_in = L_in0.index_select(dim=1, index=self.ind_i)
            self.operator_test.append(L_in)

    def _build_rhs_constant(self, x, training=True):
        ex_f = self.pde.train_aux_vars if training else self.pde.test_aux_vars
        ex_f = torch.tensor(ex_f, dtype=torch.float32, device=self.device)
        f_in = ex_f.index_select(dim=0, index=self.ind_i)
        if len(f_in.shape) == 1:
            f_in = f_in.unsqueeze(-1)
        return f_in

    def _losses(self, outputs, loss_fn, inputs, model, num_func, training):
        if training:
            operators = self.operator_train
            f_const = self.f_const
        else:
            if len(self.operator_test) == 0:
                self._build_pde_operators_test()
            operators = self.operator_test
            f_const = self._build_rhs_constant(self.pde.test_x, training=False)

        pde_losses = []
        for i in range(num_func):
            u_i = outputs[i].to(torch.float32).unsqueeze(-1).to(self.device)
            u_in = u_i.index_select(dim=0, index=self.ind_i)
            r_in = operators[i].mm(u_in) - f_const
            u_b = u_i[self.mask, :]
            zeros_b = torch.zeros((u_b.shape[0], 1), dtype=u_i.dtype, device=self.device)
            r_i = torch.cat([r_in, zeros_b], dim=0) 
            pde_losses.append(loss_fn(torch.zeros_like(r_i), r_i))
            
        pde_loss = bkd.reduce_mean(bkd.stack(pde_losses, 0))

        num_func_bc = self.bc.values.shape[0]
        bc_values = torch.tensor(self.bc.values, dtype=torch.float32, device=self.device)
        bc_loss = loss_fn(bc_values, outputs[:num_func_bc])
        return [pde_loss, bc_loss]

    def losses_train(self, targets, outputs, loss_fn, inputs, model, aux=None):
        if self.anchors is not None:
            num_func = (
                self.num_func + len(self.anchors[0])
                if self.batch_size is None
                else self.batch_size
            )
        else:
            num_func = self.num_func if self.batch_size is None else self.batch_size
        return self._losses(outputs, loss_fn, inputs, model, num_func, True)

    def losses_test(self, targets, outputs, loss_fn, inputs, model, aux=None):
        if self.anchors_test is None or self.anchors_test_y is None:
            return [torch.tensor(0.0), torch.tensor(0.0)]
        loss = loss_fn(torch.tensor(self.anchors_test_y, dtype=torch.float32, device=outputs.device), outputs)
        return [0.0*loss, loss]

    def train_next_batch(self, batch_size=None):
        if self.train_x is None:
            func_feats = self.func_space.random(self.num_func)
            func_vals = self.func_space.eval_batch(func_feats, self.eval_pts)
            vx = self.func_space.eval_batch(func_feats, self.pde.train_x[:, self.func_vars])
            if self.anchors is not None:
                anchor_kappa_vals = self.anchors[0]
                func_vals = np.vstack((anchor_kappa_vals, func_vals))
                vx_ob = np.loadtxt(f"{dname}/kappast_train.txt", delimiter=",")[: len(self.anchors[0]), :]
                vx = np.vstack((vx_ob, vx))
            self.train_x = (func_vals.astype(np.float32), self.pde.train_x)
            self.train_aux_vars = vx.astype(np.float32)
        if self.batch_size is None:
            return self.train_x, self.train_y, self.train_aux_vars

    @run_if_all_none("test_x", "test_y", "test_aux_vars")
    def test(self):
        if self.num_test is None and (self.anchors_test is None):
            self.test_x = self.train_x
            self.test_aux_vars = self.train_aux_vars
        else:
            if self.num_test is None:
                self.num_test = 0
            func_feats = self.func_space.random(self.num_test)
            func_vals = self.func_space.eval_batch(func_feats, self.eval_pts)
            vx = self.func_space.eval_batch(func_feats, self.pde.test_x[:, self.func_vars])
            if self.anchors_test is not None:
                anchor_kappa_vals = self.anchors_test[0]
                func_vals = np.vstack((anchor_kappa_vals, func_vals))
                vx_ob = np.loadtxt(f"{dname}/kappast_test.txt", delimiter=",")[: len(anchor_kappa_vals), :]
                vx = np.vstack((vx_ob, vx))
            self.test_x = (func_vals.astype(np.float32), self.pde.test_x)
            self.test_aux_vars = vx.astype(np.float32)
            self.test_y = self.anchors_test_y
        return self.test_x, self.test_y, self.test_aux_vars

class KappaFuncSpace(function_spaces.FunctionSpace):
    """
    Self-defined function space for kappa.
    We define kappa to be a linear function in (x,y):
         kappa(x,y) = a*x + b*y + c,
    where a, b are sampled randomly (and c may include randomness) so that kappa is positive.
    """
    def __init__(self, M=1, type="Linear"):
        self.type = type
        self.M = M
        self.x_max = 3

    def random(self, size):
        return np.random.rand(size, 2)

    def eval_one(self, feature, x):
        c = self.x_max * feature[0] + self.x_max * feature[1] + np.random.rand(1)
        return np.dot(feature, x.T) + c

    def eval_batch(self, features, xs):
        if xs.shape[-1] == 3:
            xs = xs[:, :2]
        Nk, Nkx = len(features), len(xs)
        c = np.repeat(
            np.sum(3 * features, axis=1).reshape((-1, 1)), Nkx, axis=1
        ) + np.tile(np.random.rand(1), (Nk, Nkx))
        return np.dot(features, xs.T) + c