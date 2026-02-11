import time
import numpy as np
import deepxde as dde
from deepxde.backend import tf
from dm_utils_tf import PDEOperatorDMCartesianProd, KappaFuncSpace
#dde.backend.set_default_backend('tensorflow.compat.v1')
dde.config.disable_xla_jit()
dname = "../../data/data_nonlinear/"

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
    # print(fname)
    X_train_1, X_train_2, y_train = d["X_train_branch"], d["X_train_trunk"], d["y_train"]
    X_test = (X_train_1[:m, :], X_train_2[:n, :])
    return X_test, y_train[:m, :n]

def load_data():
    """
    Load and return entire dataset for both training and testing.
    """
    # load data
    d = np.load("{}/nonlinear_torus_train_dm.npz".format(dname), allow_pickle=True)
    X_train_1, X_train_2, y_train = d["X_train_branch"], d["X_train_trunk"], d["y_train"]
    X_train = (X_train_1, X_train_2)

    d = np.load("{}/nonlinear_torus_test_dm.npz".format(dname), allow_pickle=True)
    X_test_1, X_test_2, y_test = d["X_test_branch"], d["X_test_trunk"], d["y_test"]
    X_test = (X_test_1, X_test_2)
    return X_train, y_train, X_test, y_test

# PDE
def transform_xyz(X):
    """
    Express (𝜃,𝜙) of a torus using (x,y,z).
    https://math.stackexchange.com/questions/4048510/express-theta-phi-of-a-torus-using-x-y-z 
    """
    R,r = 2, 1
    phis, thetas = [], []
    for i in range(len(X)):
        x, y, z = X[i, 0], X[i, 1], X[i, 2]
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
        const_sqrt = tf.constant(np.sqrt(2 / np.pi), dtype=x.dtype)
        return 0.5 * x * (1 + tf.math.tanh(const_sqrt * (x + 0.044715 * x ** 3)))

    def pde(x, y, L_mat_current, kappa):
        """-div_g(κgrad_g(u)) + u = f --> (-diag(kappa)*L + I)u - f = 0
           Inputs: 
            kappa: the input function
            L_mat: matrix constructed
            f is defined as an auxiliary function"""
        indices = L_mat_current.indices
        values = L_mat_current.values
        dense_shape = L_mat_current.dense_shape
        kappa_reshaped = tf.reshape(kappa, (-1, 1))
        neg_sparse_mat = tf.SparseTensor(indices=indices, values=-values, dense_shape=dense_shape)
        f = 3/2 * tf.square(y) + 2 * kappa_reshaped * y - 1/2 * tf.square(kappa_reshaped) + y 
        eye_sparse = tf.sparse.eye(2500, dtype=tf.float32)
        sparse_mat = tf.sparse.add(neg_sparse_mat, eye_sparse)
        res = tf.sparse.sparse_dense_matmul(sparse_mat, y) - f
        return res    

    ## Load Data
    # Load entire dataseta
    X_train, y_train, X_test, y_test = load_data()
    # Load data for the PDE residual loss term
    N_k, N_p = 100,2500
    anchors_train = load_train_data("{}/nonlinear_torus_train_dm.npz".format(dname), N_k, N_p)[0]
    # Load data for testing
    N_kt, N_pt = 2,2500
    anchors_test_data = load_test_data("{}/nonlinear_torus_test_dm.npz".format(dname), N_kt, N_pt)
    # Load data for the observation loss
    N_ob_k, N_ob_p = 2,2500
    anchors_train_ob,  anchors_train_ob_y = load_train_data("{}/nonlinear_torus_train_dm.npz".format(dname), N_ob_k, N_ob_p)

    # Construct pointcloud as the domain
    geom = dde.geometry.pointcloud.PointCloud(X_train[1])
    observe_y = dde.icbc.PointSetBC(anchors_train_ob[1], anchors_train_ob_y, component=0, shuffle=False)
    pde = dde.data.PDE(geom, pde, [observe_y], num_domain=0)

    func_space = KappaFuncSpace() # kappa should be in the same form as oberservations
    x = np.linspace(-3, 3, num=26) # domain is -3,3 
    y = np.linspace(-3, 3, num=26) # Nkx = 26*26
    xv, yv = np.meshgrid(x, y)
    eval_pts = np.vstack((np.ravel(xv.T), np.ravel(yv.T))).T # for branch net

    # N_func, function_variables: kappa is only a function of x and y, 
    # num_test is the number of function for testing
    print("Running PDEOperatorDMCartesianProd...")
    data = PDEOperatorDMCartesianProd(pde, func_space, eval_pts, num_function = 0, anchors = anchors_train, anchors_test_data = anchors_test_data, function_variables=[0, 1], batch_size=None)

    # # Save data
    # np.savetxt("dm_data/dm_train_kappa_ob_{}.txt".format(fn), data.train_x[0], delimiter=',')
    # np.savetxt("dm_data/dm_train_kappa_x_ob_{}.txt".format(fn), data.train_aux_vars, delimiter=',')
    # np.savetxt("dm_data/dm_train_x_ob_{}.txt".format(fn), data.train_x[1], delimiter=',')

    m = 26 ** 2
    n = 32
    activation = "relu"
    branch_net = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(m,)),
            tf.keras.layers.Reshape((26, 26, 1)),
            tf.keras.layers.Conv2D(8, (3, 3), strides=2, activation=activation),
            tf.keras.layers.Conv2D(16, (3, 3), strides=2, activation=activation),
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
    # Compile and Train
    model.compile("adam", lr=0.001, decay=("inverse time",20000, 0.5), loss_weights = [0.001, 1])  #, metrics=["l2 relative error"])
    t1 = time.time()
    losshistory, train_state = model.train(iterations=100000, batch_size=None)#, model_save_path="model/model.ckpt")
    t2 = time.time()
    print("Time of training:", t2 - t1)

    # Save results and Plot the loss trajectory
    dde.utils.save_loss_history(losshistory, "dm_data/loss_ob_{}.dat".format(fn))
    dde.utils.plot_loss_history(losshistory, "dm_data/loss_ob_{}.png".format(fn))


    # predict
    pred_anchors = model.predict(anchors_test_data[0])
    print(dde.metrics.mean_squared_error(anchors_test_data[1].reshape((-1,1)), pred_anchors.reshape((-1,1))))
    print(dde.metrics.l2_relative_error(anchors_test_data[1].reshape((-1,1)), pred_anchors.reshape((-1,1))))
    pred_u = model.predict(X_test)
    print("Mean squared error of testing data: ",dde.metrics.mean_squared_error(y_test.reshape((-1,1)), pred_u.reshape((-1,1))))
    print("L2 relative error of testing data: ", dde.metrics.l2_relative_error(y_test.reshape((-1,1)), pred_u.reshape((-1,1))))
    np.savetxt("dm_data/dm_pred_u_ob_{}.txt".format(fn), pred_u, delimiter=',')

if __name__ == "__main__":
    main("2")
