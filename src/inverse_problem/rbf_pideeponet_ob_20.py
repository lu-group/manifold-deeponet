import os
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from deepxde.backend import tf
from rbf_utils_20 import PDEOperatorRBFCartesianProd, KappaFuncSpace
dde.config.disable_xla_jit()

m=20
sname = "_rbf"
dname = "../../data/data_inverse/data_20_5000"
fname_train = f"{dname}/inverse_torus_train{sname}.npz"
fname_test = f"{dname}/inverse_torus_test{sname}.npz"

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
        a,b = 1, 1
        x = np.loadtxt(f"{dname}/tp_train.txt", delimiter=",")
        t,p = x[:,0:1], x[:,1:2]
        sin_ax, cos_ax = np.sin(a*t), np.cos(a*t)
        sin_by, cos_by = np.sin(b*p), np.cos(b*p)
        sin_x, cos_x = np.sin(t), np.cos(p)
        u = sin_ax * sin_by
        u_x, u_y = a * cos_ax * sin_by, b * sin_ax * cos_by
        u_xx, u_yy = -a**2 * u, -b**2 * u
        c = 2 + u
        c_x, c_y = a * cos_ax * sin_by, b * sin_ax * cos_by
        ginv1 = np.ones((m**2, 1))
        ginv2 = 1./(2+np.cos(t))**2
        ginv = np.hstack((ginv1, ginv2))
        Gamma1_22 = sin_x * (2 + cos_x)
        f = - (ginv[:, 0:1] * c_x * u_x + ginv[:, 1:2] * c_y * u_y + \
           c * (ginv[:, 0:1] * u_xx + ginv[:, 1:2] * u_yy - \
                ginv[:, 1:2] * Gamma1_22 * u_x)) + u
        return f

    def pde(x, y, kappa, Gx, Gy, Gz, ex_f):
        """-div_g(κgrad_g(u)) + u = f --> weak form $1/N \sum(G_x(\kappa(G_x u))+G_y\kappa(G_yu)+G_z(\kappa(G_zu)))-1/N\sum(u^2) - 1/N\sum(f.u)$ 
           Inputs: 
            kappa: the input function
            f is defined as an auxiliary function"""
        N = m**2
        kappa_reshaped = tf.reshape(kappa, (N,1))
        G_components = [Gx, Gy, Gz]
        gg_components = [tf.math.multiply(tf.matmul(G, kappa_reshaped), G) for G in G_components]
        gg_sum = tf.add_n(gg_components)
        kappa_sum = tf.add_n([tf.matmul(G, G) for G in G_components])
        L_mat = -(gg_sum) - tf.math.multiply(kappa_reshaped, kappa_sum) + tf.eye(N, dtype=tf.float32)
        res = tf.matmul(L_mat, y) - ex_f
        return res

    ## Load Data
    # Load entire dataset
    X_train, y_train, X_test, y_test = load_data()
    # Load data for the PDE residual loss term
    N_k, N_p = 1000,m**2
    anchors_train = load_train_data(fname_train, N_k, N_p)[0]
    # Load data for testing
    N_kt, N_pt = 100,m**2
    anchors_test_data = load_test_data(fname_test, N_kt, N_pt)
    # Load data for the observation loss
    N_ob_k, N_ob_p = 1000,m**2
    anchors_train_ob,  anchors_train_ob_y = load_train_data(fname_train, N_ob_k, N_ob_p)
    
    # Construct pointcloud as the domain
    geom = dde.geometry.PointCloud(X_train[1])
    observe_y = dde.icbc.PointSetBC(anchors_train_ob[1], anchors_train_ob_y, component=0, shuffle=False)
    pde = dde.data.PDE(geom, pde, [observe_y], num_domain=0, auxiliary_var_function=ex_func)

    func_space = KappaFuncSpace() # kappa should be in the same form as oberservations
    x = np.linspace(-np.pi, np.pi, num=m) # domain is -3,3 
    y = np.linspace(-np.pi, np.pi, num=m) # Nkx = 26*26
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
    n = 32
    activation = "relu"
    branch_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(m**2,)),
            tf.keras.layers.Reshape((m, m, 1)),
            tf.keras.layers.Conv2D(16, (3, 3), strides=2, activation=activation),
            tf.keras.layers.Conv2D(32, (3, 3), strides=2, activation=activation),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation=activation),
            tf.keras.layers.Dense(n),
        ]
    )
    branch_net.summary()

    # network
    net = dde.nn.DeepONetCartesianProd(
            [m**2, branch_net],
            [3, n, n, n],
            {"branch": "relu", "trunk": gelu},
            "Glorot normal",
        )

    # model
    model = dde.Model(data, net)
    # Compile and Train
    model.compile("adam", lr=0.001, decay=("inverse time",20000, 0.5), loss_weights = [0.005, 1])  #,metrics=["l2 relative error"])
    losshistory, train_state = model.train(iterations=200002, batch_size=None, display_every = 1000, model_save_path="model/model1000")

    pred_u = model.predict(X_test)
    print("Mean squared error of testing data: ",dde.metrics.mean_squared_error(y_test.reshape((-1,1)), pred_u.reshape((-1,1))))
    print("L2 relative error of testing data: ", dde.metrics.l2_relative_error(y_test.reshape((-1,1)), pred_u.reshape((-1,1))))
    os.makedirs("rbf_res", exist_ok=True)
    np.savetxt(f"rbf_res/rbf_pred_u_20.txt", pred_u, delimiter=',')

if __name__ == "__main__":
    main("1000_2")
