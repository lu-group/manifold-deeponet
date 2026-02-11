import numpy as np
import time
import deepxde.backend as bkd
from deepxde.backend import tf
from deepxde.data import Data, function_spaces
from deepxde.data.sampler import BatchSampler
from deepxde.utils import run_if_all_none, array_ops_compat

dname = "../../data/data_inverse/data_50/"

class PDEOperatorRBFCartesianProd(Data):

    def __init__(
        self,
        pde,
        function_space,
        evaluation_points,
        num_function,
        anchors = None,
        anchors_test_data = None,
        function_variables=None,
        num_test=None,
        batch_size=None,
    ):
        
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
        self.anchors_test_y = anchors_test_data[1]
        self.anchors_test = anchors_test_data[0]

        self.train_x = None
        self.train_y = None
        self.train_aux_vars = None
        self.test_x = None
        self.test_y = None
        self.test_aux_vars = None

        self.Gx_train, self.Gy_train, self.Gz_train = self.get_G_matrix() 
        self.Gx_test, self.Gy_test, self.Gz_test = self.get_G_matrix()

        self.train_next_batch()
        self.test()

    def _losses(self, outputs, loss_fn, inputs, model, num_func, training):    # only run when compiling, tensor, inputs:self.net.inputs
        beg, end = 0, self.pde.num_bcs[0] # 0, 2500
        losses = []
        W_ob = 1 # weights for observations
        W_pde = 1
        bc = self.pde.bcs[0]
        num_func_bc = bc.values.shape[0]

        print("num", num_func, num_func_bc, inputs, len(self.pde.train_x_all), len(self.pde.test_x), self.pde.test_aux_vars.shape, self.pde.train_aux_vars.shape)
        aux_vars = tf.cast(self.train_aux_vars if training else self.test_aux_vars, tf.float32)
        Gx = self.Gx_train if training else self.Gx_test
        Gy = self.Gy_train if training else self.Gy_test
        Gz = self.Gz_train if training else self.Gz_test
        pde_aux_vars = self.pde.train_aux_vars if training else self.pde.test_aux_vars
        pde_aux_vars_sliced = pde_aux_vars[-len(self.pde.train_x_all):, :]

        bc_values = tf.convert_to_tensor(bc.values, dtype=tf.float32)
        bc_values_reshaped = tf.reshape(bc_values, [-1, bc.values.shape[1]])

        for i in range(num_func):
            # compute the loss for pde residual
            out = outputs[i][:, None]
            f = self.pde.pde(inputs[1], out, aux_vars[i], Gx, Gy,Gz, pde_aux_vars_sliced) if self.pde.pde is not None else []               
            f = [f] if not isinstance(f, (list, tuple)) else f            
            error_f = [fi[:] for fi in f]
            losses_i = [W_pde * loss_fn(bkd.zeros_like(error), error) for error in error_f]

            # compute the loss for observations
            if training and i < num_func_bc:
                error = tf.reshape(out[beg:end], [bc.values.shape[1],]) - bc_values_reshaped[i]                 #bc.values.shape - N_ob_k, N_ob_p
                losses_i.append(W_ob*loss_fn(bkd.zeros_like(error), error))
            losses.append(losses_i)

        losses = zip(*losses)
        losses = [bkd.reduce_mean(bkd.as_tensor(l)) for l in losses]
        #print("Time of loading:", time.time() - t0)
        return losses
    
    def losses_train(self, targets, outputs, loss_fn, inputs, model, aux=None):
        if self.anchors is not None:
            num_func = self.num_func + len(self.anchors[0]) if self.batch_size is None else self.batch_size
        else:
            num_func = self.num_func if self.batch_size is None else self.batch_size
        return self._losses(outputs, loss_fn, inputs, model, num_func, True)

    def losses_test(self, targets, outputs, loss_fn, inputs, model, aux=None):
        beg, end = 0, self.pde.num_bcs[0] # 0, 2500
        losses = []
        num_func = self.test_x[0].shape[0]
        for i in range(num_func):
            out = outputs[i][:, None]
            error = bkd.tf.reshape(out[beg:end], (self.anchors_test_y.shape[1],))- self.anchors_test_y[i]                    #bc.values.shape - N_ob_k, N_ob_p
            losses_i = [loss_fn(bkd.zeros_like(error), error)]
            losses.append(losses_i)

        losses = zip(*losses)
        losses = [bkd.reduce_mean(bkd.as_tensor(l)) for l in losses]
        return losses
    
    def train_next_batch(self, batch_size=None):                                                     # do not use batch_size in model.train
        if self.train_x is None:
            func_feats = self.func_space.random(self.num_func)
            func_vals = self.func_space.eval_batch(func_feats, self.eval_pts)                         #(Nk, Nkx) values of kappa at eval_pts
            vx = self.func_space.eval_batch(func_feats, self.pde.train_x[:, self.func_vars])          # values of kappa at training points in the domain + anchors
            if self.anchors is not None:
                func_feats = self.func_space.random(self.num_func + len(self.anchors[0]))             
                func_vals = np.vstack((self.anchors[0],func_vals))                                    # first anchors, then sampled funcs
                vx_ob = np.loadtxt(f"{dname}/kappas_train.txt",delimiter=",")[:len(self.anchors[0]), :]
                vx = np.vstack((vx_ob, vx))
            self.train_x = (func_vals, self.pde.train_x)  # train_x_all - sampled in domain; train_x -  train_x_all + bc 
                      #(branch net, trunck net), sampled+anchors
            self.train_aux_vars = vx #(Nk, N)                                                          # self.train_aux_vars kappa on torus, self.train_x[0] kappa for branch net

        if self.batch_size is None:
            return self.train_x, self.train_y, self.train_aux_vars

    @run_if_all_none("test_x", "test_y", "test_aux_vars")
    def test(self):
        # if num_test is none, then the testing data is the training data
        if self.num_test is None and (self.anchors_test is None):
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
            if self.anchors_test is not None:
                func_feats = self.func_space.random(self.num_test + len(self.anchors_test[0]))             
                func_vals = np.vstack((self.anchors_test[0],func_vals))
                vx_ob = np.loadtxt(f"{dname}/kappas_test.txt",delimiter=",")[:len(self.anchors_test[0]), :]
                vx = np.vstack((vx_ob, vx))
            self.test_x = (func_vals, self.pde.test_x)
            
            self.test_aux_vars = vx
        return self.test_x, self.test_y, self.test_aux_vars
    
    def get_G_matrix(self):
        #print("Running get_G_matrix...")
        
        Gx_train = np.loadtxt(f"{dname}/Gx.dat", delimiter=",")
        Gy_train = np.loadtxt(f"{dname}/Gy.dat", delimiter=",")
        Gz_train = np.loadtxt(f"{dname}/Gz.dat", delimiter=",")
        Gx_train = tf.cast(Gx_train, tf.float32)
        Gy_train = tf.cast(Gy_train, tf.float32)
        Gz_train = tf.cast(Gz_train, tf.float32)
        print("Gz_train", Gz_train.shape)
        return Gx_train, Gy_train, Gz_train  # for one kappa , G: (N_trainx , N_trainx), G_train: (Nk, N_trainx , N_trainx)

    
class KappaFuncSpace(function_spaces.FunctionSpace):
    """
    Self-defined kappa.
    Kappa = ax + by + c, needs to be positive
    Args:
        M (float): `M` > 0. The coefficients a, b are randomly sampled from [-`M`, `M`].
    """
    def __init__(self, M=1, type="Linear"):
        self.type = type
        self.M = M
        self.x_max = 3
    
    def random(self, size): 
        # size: num_func, return: features (size, n_features)
        return np.random.rand(size, 2)

    def eval_one(self, feature, x):
        c = self.x_max*feature[0] + self.x_max*feature[1] + np.random.rand(1)
        return np.dot(feature,x.T)+c
    
    def eval_batch(self, features, xs):
        if xs.shape[-1] == 3:
            xs = xs = xs[:, :2]
        Nk, Nkx = len(features), len(xs)
        c = np.repeat(np.sum(np.dot(self.x_max,features), axis = 1).reshape((-1,1)), Nkx, axis=1) + np.tile(np.random.rand(1), (Nk, Nkx))
        return np.dot(features,xs.T) + c # (Nk, Nkx)




