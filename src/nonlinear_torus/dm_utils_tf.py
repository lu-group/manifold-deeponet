import numpy as np
import time 
import scipy
import deepxde.backend as bkd
from deepxde.backend import tf
from deepxde.data import Data, function_spaces
from deepxde.data.sampler import BatchSampler
from deepxde.utils import run_if_all_none
from sklearn.neighbors import NearestNeighbors

dname= "../../data/data_nonlinear/"

class PDEOperatorDMCartesianProd(Data):

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

        self.dm_train, self.dm_test = None, None
        self.L_mat_train, self.L_mat_test = None, None

        self.train_next_batch()
        self.test()

    def _losses(self, outputs, loss_fn, inputs, model, num_func, training):    # only run when compiling, tensor, inputs:self.net.inputs
        beg, end = 0, self.pde.num_bcs[0] # 0, 2500
        W_ob = 1 # weights for observations
        W_pde = 1
        bc = self.pde.bcs[0]
        num_func_bc = bc.values.shape[0]
        self.train_aux_vars  = np.float32(self.train_aux_vars)
        L_mat = [self.get_L_matrix(self.train_aux_vars[i] if training else self.test_aux_vars[i], True if training else False) \
                 for i in range(num_func)]
        losses = []
        for i in range(num_func):
            # compute the loss for pde residual
            out = outputs[i][:, None]
            L_mat_current = L_mat[i] 
            kappa = self.train_aux_vars[i] if training else self.test_aux_vars[i]
            f = self.pde.pde(inputs[1], out, L_mat_current, kappa)  if self.pde.pde is not None else []  
            f = [f] if not isinstance(f, (list, tuple)) else f
                
            error_f = [fi[:] for fi in f]
            losses_i = [W_pde * loss_fn(bkd.zeros_like(error), error) for error in error_f]

            # compute the loss for observations
            if i < num_func_bc:
                error = bkd.tf.reshape(out[beg:end], (bc.values.shape[1],))- bc.values[i]                    #bc.values.shape - N_ob_k, N_ob_p
                losses_i.append(W_ob*loss_fn(bkd.zeros_like(error), error))
            else:
                losses_i.append(W_ob*loss_fn(bkd.zeros_like(error), bkd.zeros_like(error)))
            losses.append(losses_i)
        losses = zip(*losses)
        losses = [bkd.reduce_mean(bkd.as_tensor(l)) for l in losses]
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
            func_vals = self.func_space.eval_batch(func_feats, self.eval_pts)                         # (Nk, Nkx) values of kappa at eval_pts
            vx = self.func_space.eval_batch(func_feats, self.pde.train_x[:, self.func_vars])          # values of kappa at training points in the domain + anchors
            if self.anchors is not None:
                func_feats = self.func_space.random(self.num_func + len(self.anchors[0]))             
                func_vals = np.vstack((self.anchors[0],func_vals))                                    # first anchors, then sampled funcs
                vx_ob = np.loadtxt("{}kappast_train.txt".format(dname),delimiter=",")[:len(self.anchors[0]), :]
                vx = np.vstack((vx_ob, vx))
            self.train_x = (func_vals, self.pde.train_x)  # train_x_all - sampled in domain; train_x -  train_x_all + bc           #(branch net, trunck net), sampled+anchors
            self.train_aux_vars = vx #(Nk, N)                                                          # self.train_aux_vars kappa on torus, self.train_x[0] kappa for branch net

        if self.batch_size is None:
            if self.dm_train is None:
                self.dm_train = DiffusionMap(self.train_x[1], self.train_aux_vars) # X, kappa
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
                vx_ob = np.loadtxt("{}kappast_test.txt".format(dname),delimiter=",")[:len(self.anchors_test[0]), :]
                vx = np.vstack((vx_ob, vx))
            self.test_x = (func_vals, self.pde.test_x)
            
            self.test_aux_vars = vx
            if self.dm_test is None:
                self.dm_test = DiffusionMap(self.test_x[1], self.test_aux_vars)
        return self.test_x, self.test_y, self.test_aux_vars
    
    def get_L_matrix(self, kappa_i, training):
        #print("Running get_L_matrix...")
        if training:
            L_mat = self.dm_train.get_matrix(kappa_i)
        else:
            L_mat = self.dm_test.get_matrix(kappa_i)
        
        L_mat = self.convert_sparse_matrix_to_sparse_tensor(L_mat)
        return L_mat
    
    def convert_sparse_matrix_to_sparse_tensor(self,X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)
    
class DiffusionMap:
    def __init__(self, X, kappa):
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
        # automated pre-tuning of epsilon
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
        #return np.array(np.tile([0.4229,0.0942], (size,1)))

    def eval_one(self, feature, x):
        c = self.x_max*feature[0] + self.x_max*feature[1] + np.random.rand(1)
        return np.dot(feature,x.T)+c
    
    def eval_batch(self, features, xs):
        if xs.shape[-1] == 3:
            xs = xs[:, :2]
        Nk, Nkx = len(features), len(xs)
        c = np.repeat(np.sum(np.dot(self.x_max,features), axis = 1).reshape((-1,1)), Nkx, axis=1) + np.tile(np.random.rand(1), (Nk, Nkx))
        
        return np.dot(features,xs.T) + c # (Nk, Nkx)
