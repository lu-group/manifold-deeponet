# 1 linear 
import time
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from deepxde.backend import tf
from gmls_utils_tf import PDEOperatorRBFCartesianProd, KappaFuncSpace
dde.config.set_default_float("float32")
dde.config.disable_xla_jit()

path = "../../data/data_semi/"
fname_train = f"{path}semitorus_data_train1_rbf.npz"
fname_test = f"{path}semitorus_data_test1_rbf.npz"

def load_test_data(fname, m, n):
    """
    Load data for testing.
    Inputs: fname - file name
    m - number of kappas
    n - number of collocation points
    """

    if m == 0 and n == 0:
        return None, None
    # load data
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
    # load data
    d = np.load(fname, allow_pickle=True)
    X_train_1, X_train_2, y_train = d["X_train_branch"], d["X_train_trunk"], d["y_train"]
    X_test = (X_train_1[:m, :], X_train_2[:n, :])
    return X_test, y_train[:m, :n]

def load_data():
    """
    Load and return entire dataset for both training and testing.
    """
    # load data
    d = np.load(fname_train, allow_pickle=True)
    X_train_1, X_train_2, y_train = d["X_train_branch"], d["X_train_trunk"], d["y_train"]
    X_train = (X_train_1, X_train_2)

    d = np.load(fname_test, allow_pickle=True)
    X_test_1, X_test_2, y_test = d["X_test_branch"], d["X_test_trunk"], d["y_test"]
    X_test = (X_test_1, X_test_2)
    return X_train, y_train, X_test, y_test

# PDE
def transform_xyz(X):
    # https://math.stackexchange.com/questions/4048510/express-theta-phi-of-a-torus-using-x-y-z 
    R,r = 2, 1
    phis, thetas = [], []
    for i in range(len(X)):
        x, y, z = X[i, 0], X[i, 1], X[i, 2]
        # compute theta
        theta = np.arcsin(z/r)
        tmp = np.arcsin(y/np.sqrt(x**2 + y**2))
        if x**2 + y**2 < R**2:
            theta = np.pi - np.arcsin(z)
        elif z < 0:
            theta = np.pi*2 + np.arcsin(z)
        else:
            theta =  np.arcsin(z)
        if x < 0:
            phi = np.pi -  tmp
        elif y < 0:
            phi = 2*np.pi +  tmp
        else:
            phi = tmp
        phis.append(phi)
        thetas.append(theta)
    return phis, thetas

def main(fn):

    def gelu(x):
        return 0.5 * x * (1 + tf.math.tanh(tf.math.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


    def ex_func(inputs):
        """Extra function in PDE: Source term f.
            Input: inputs, array of (x,y,z), shape: (N,3)
            Output: f. shape: (N,3). N is the number of points
        """
        # source term in the pde
        p, t = transform_xyz(inputs)
        p,t = np.array(p).reshape((-1,1)), np.array(t).reshape((-1,1))
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
        return f

    def pde(x, y, kappa, Gx, Gy, Gz, Lap, ind_i, ex_f):
        """-div_g(κgrad_g(u)) + u = f --> weak form $1/N \sum(G_x(\kappa(G_x u))+G_y\kappa(G_yu)+G_z(\kappa(G_zu)))-1/N\sum(u^2) - 1/N\sum(f.u)$ 
           Inputs: 
            kappa: the input function
            f is defined as an auxiliary function"""
        N = 2500
        kappa_reshaped = tf.cast(tf.reshape(kappa, (N,1)),  tf.float32)
        G_components = [Gx, Gy, Gz]
        gg_components = [tf.math.multiply(tf.matmul(G, kappa_reshaped), G) for G in G_components]
        gg_sum = tf.add_n(gg_components)
        L_mat = -(gg_sum) - tf.math.multiply(kappa_reshaped, Lap) + tf.eye(N, dtype=tf.float32)
        #Matlab: L=-((Gx*kappa).*Gx+(Gy*kappa).*Gy+(Gz*kappa).*Gz+kappa.*Lap)+eye(N);
        ind_i = [[int(i)-1] for i in ind_i]
        L_in0 = tf.gather_nd(L_mat, indices=ind_i)
        ind_i = [i[0] for i in ind_i]
        L_in = tf.gather(L_in0,indices=ind_i, axis = 1) #  shape=(2433, 2433)
        ind_b = [i for i in range(1, 2500) if i not in ind_i]
        u_in = tf.gather(y,indices=ind_i, axis = 0)
        u_b = tf.gather(y,indices=ind_b, axis = 0)
        ex_f = tf.gather(ex_f,indices=ind_i, axis = 0)
        res = tf.matmul(L_in, u_in) - ex_f
        res = tf.concat([res, tf.zeros([u_b.shape[0], 1], dtype = tf.float32)], 0)
        return res

    ## Load Data
    # Load entire dataset
    X_train, _, X_test, y_test = load_data()
    # Load data for the PDE residual loss term
    N_k, N_p = 2,2500
    anchors_train = load_train_data(fname_train, N_k, N_p)[0]
    # Load data for testing
    N_kt, N_pt = 2,2500
    anchors_test_data = load_test_data(fname_test, N_kt, N_pt)
    # Load data for the observation loss
    N_ob_k, N_ob_p = 2, 2500
    anchors_train_ob,  anchors_train_ob_y = load_train_data(fname_train, N_ob_k, N_ob_p)

    # Construct pointcloud as the domain
    geom = dde.geometry.PointCloud(X_train[1]) 
    observe_y = dde.icbc.PointSetBC(anchors_train_ob[1], anchors_train_ob_y, component=0, shuffle=False)
    pde = dde.data.PDE(geom, pde, [observe_y], num_domain=0, auxiliary_var_function=ex_func)

    func_space = KappaFuncSpace() # kappa should be in the same form as oberservations
    x = np.linspace(-3, 3, num=26) # domain is -3,3 
    y = np.linspace(-3, 3, num=26) # Nkx = 26*26
    xv, yv = np.meshgrid(x, y)
    eval_pts = np.vstack((np.ravel(xv.T), np.ravel(yv.T))).T # for branch net

    # N_func, function_variables: kappa is only a function of x and y, 
    # num_test is the number of function for testing
    print("Running PDEOperatorRBFCartesianProd...")
    data = PDEOperatorRBFCartesianProd(pde, func_space, eval_pts, num_function = 0, anchors = anchors_train, anchors_test_data = anchors_test_data, function_variables=[0, 1], batch_size=None)
    
    # # Save data
    # np.savetxt("rbf_data/rbf_train_kappa_{}.txt".format(fn), data.train_x[0], delimiter=',')
    # np.savetxt("rbf_data/rbf_train_kappa_x_{}.txt".format(fn), data.train_aux_vars, delimiter=',')
    # np.savetxt("rbf_data/rbf_train_x_{}.txt".format(fn), data.train_x[1], delimiter=',')

    # Net
    # CNN for branch net
    m = 26 ** 2
    n = 32
    activation = "relu"
    branch_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(m,)),
            tf.keras.layers.Reshape((26, 26, 1)),
            tf.keras.layers.Conv2D(16, (5, 5), strides=2, activation=activation),
            tf.keras.layers.Conv2D(32, (5, 5), strides=2, activation=activation),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(n, activation=activation),
            tf.keras.layers.Dense(n),
        ]
    )
    branch_net.summary()

    # network
    net = dde.nn.DeepONetCartesianProd(
            [m, branch_net],
            [3, n, n, n],
            {"branch": "relu", "trunk": gelu},
            "Glorot normal",
        )

    # model
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, decay=("inverse time",20000, 0.5), loss_weights = [1e-6, 1])  #, metrics=["l2 relative error"]) # 0.00001
    t1 = time.time()
    losshistory, train_state = model.train(iterations=5000, batch_size=None)#, model_save_path="model/model.ckpt")
    t2 = time.time()
    print("Time of training:", t2 - t1)

    # Save results and Plot the loss trajectory
    dde.utils.save_loss_history(losshistory, "res_linear/loss_{}.dat".format(fn))
    dde.utils.plot_loss_history(losshistory, "res_linear/loss_{}.png".format(fn))

    
    # predict
    pred_anchors = model.predict(anchors_test_data[0])
    print(dde.metrics.mean_squared_error(anchors_test_data[1].reshape((-1,1)), pred_anchors.reshape((-1,1))))
    print(dde.metrics.l2_relative_error(anchors_test_data[1].reshape((-1,1)), pred_anchors.reshape((-1,1))))
    
    pred_u = model.predict(X_test)
    print("Mean squared error of testing data: ",dde.metrics.mean_squared_error(y_test.reshape((-1,1)), pred_u.reshape((-1,1))))
    print("L2 relative error of testing data: ", dde.metrics.l2_relative_error(y_test.reshape((-1,1)), pred_u.reshape((-1,1))))
    np.savetxt("res_linear/rbf_pred_u_{}.txt".format(fn), pred_u, delimiter=',')


main("100_10_1")
