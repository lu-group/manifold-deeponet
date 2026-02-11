import time
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
import scipy.linalg as la
import scipy.sparse.linalg as sla
from sklearn.neighbors import NearestNeighbors
from pideeponet_predict_20 import predict_model, predict_u

def GL(X, K, KNN):
    N = X.shape[0]
    dist=np.zeros((N,N))
    for i in range(N):
        for j in range(i, N):
            dist[i,j] = np.linalg.norm(X[i, :]-X[j, :])
    dist = dist + dist.transpose()
    
    tau=np.zeros((N,1))
    for i in range(N):
        sorted = np.sort(dist[i, :])
        tau[i] = sorted[K]
    
    W = np.zeros((N, N))
    if KNN==1:
        for i in range(N):
            for j in range(i, N):
                if dist[i,j] <= max(tau[i], tau[j]):
                    W[i, j] = np.exp(-dist[i, j]**2/(tau[i]*tau[j]*2))
    else:
        for i in range(N):
            for j in range(i+1, N):
                W[i, j] = np.exp(-dist[i, j]**2/(tau[i]*tau[j]*2))
    W = W + W.transpose()
    for i in range(N):
        W[i,i] = 1
    DD = np.zeros((N,N))
    for i in range(N):
        DD[i, i] = 1/np.sqrt(np.sum(W[i, :], axis=0))
    
    L=np.round((np.eye(N)-DD@W@DD)/np.mean(tau)**2,10); 
    return L, W, dist

def knn(X, k):
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nbrs.kneighbors(X) #distances.shape (N, k), indices.shape (N,k)
    return distances, indices

def estimate_epsilon(N,k,d, epss):
    # automated pre-tuning of epsilon
    dpreGlobal = []
    for ll in range(len(epss)):
        dpreGlobali = np.sum(np.sum(np.exp(-d**2/(4*epss[ll])), axis = 0).reshape((1,-1)), axis = 1)/(N*k)
        dpreGlobal.append(dpreGlobali)
    dpreGlobal = np.array(dpreGlobal).reshape((1,-1)) # (1, 401)
    halfdim = np.diff(np.log(dpreGlobal))/np.diff(np.log(epss).reshape((1, -1))) # (1,400)
    maxval,maxind = max(halfdim), halfdim.argmax()
    epsilon = epss[maxind]
    return epsilon

def to_sparse(i,j,v,m,n):
        return scipy.sparse.csr_matrix((v, (i, j)), shape=(m, n))

def BEIP_T_DeepONet(m,NN,sigma,beta,tau,gamma,obs_set,Nexp, M, is_orig):
    """
    Inputs:
      m = number of points on manifold
      NN = number of neighbors when constructing graph Laplacian
      sigma = noise level
      beta = step size of pCN
      tau = the parameter tau in the prior
      gamma = the parameter s in the prior
      obs_set = the index set for the observation locations
      Nexp = counter of experiments (because I need to run this for different paramters.) 
    Outputs:
      accept = number of accepted samples (use this to tune beta)
      PM = posterior mean of theta (log diffusion coefficient)
      pm_u = 97.5% posterior quantile 
      pm_l = 2.5% posterior quantile 
      p_pm = posterior mean of solution
      c = true diffusion coefficient
      u = true solution 
      e = error
    """
    # Generate data x, y
    dx2 = 2*np.pi/m
    test2 = np.linspace(dx2/2, 2*np.pi-dx2/2, m)
    X, Y = np.meshgrid(test2, test2)
    e = np.zeros((3,1))
    N = m**2
    #x - w1 and w2, y - X Y Z, 
    x1 = np.reshape(X, (N, 1), order='F')
    x2 = np.reshape(Y, (N, 1), order='F')
    x = np.hstack((x1, x2))
    y1 = np.multiply(2+np.cos(x[:,0:1]), np.cos(x[:,1:2]))
    y2 = np.multiply(2+np.cos(x[:,0:1]), np.sin(x[:,1:2]))
    y3 = np.sin(x[:,0:1])
    y = np.hstack((y1, y2, y3))

    #find an appropirate epsilon 
    #k=math.floor(0.2*N)
    k=math.floor(1.5*math.sqrt(N))
    d=2
    alpha=1/2
    epss=[2**i for i in np.linspace(-30,10,401)]
    distances, indices = knn(y, k)
    epsilon = estimate_epsilon(N, k, distances, epss)

    W0 = np.exp(-distances**2/(4*epsilon)) # (N,k)
    i = np.transpose(indices).reshape((N*k, ), order = "F")
    j = np.tile(range(0,N), (k, 1)).reshape((N*k, ), order = "F")
    v = np.transpose(W0).reshape((N*k, ), order = "F")
    W = np.transpose(to_sparse(i, j, v, N, N))

    #u = sin(w1)*sin(w2), kappa /c = sin(w1)*sin(w2)
    a,b = 1, 1
    u = np.multiply(np.sin(a*x[:,0:1]), np.sin(b*x[:,1:2]))
    u_x = a*np.multiply(np.cos(a*x[:,0:1]), np.sin(b*x[:,1:2]))
    u_y = b*np.multiply(np.sin(a*x[:,0:1]), np.cos(b*x[:,1:2]))
    u_xx = -a**2*np.multiply(np.sin(a*x[:,0:1]), np.sin(b*x[:,1:2]))
    u_yy = -b**2*np.multiply(np.sin(a*x[:,0:1]), np.sin(b*x[:,1:2]))
    c = 2 + np.multiply(np.sin(a*x[:,0:1]), np.sin(b*x[:,1:2]))
    c_x = a*np.multiply(np.cos(a*x[:,0:1]), np.sin(b*x[:,1:2]))
    c_y = b*np.multiply(np.sin(a*x[:,0:1]), np.cos(b*x[:,1:2]))

    #Riemannian metric
    ginv1 = np.ones((N, 1))
    ginv2 = 1./(2+np.cos(x[:,0:1]))**2
    ginv = np.hstack((ginv1, ginv2))
    rdetginv = 1./(2+np.cos(x[:,0:1]))
    rdetg_x = -np.sin(x[:,0:1])
    Gamma1_22 = np.multiply(np.sin(x[:,0:1]), 2 + np.cos(x[:,0:1]))
    f =  -(np.multiply(np.multiply(ginv[:,0:1], c_x), u_x) + \
        np.multiply(np.multiply(ginv[:,1:2], c_y), u_y) + \
        np.multiply(c, np.multiply(ginv[:,0:1], u_xx) + np.multiply(ginv[:,1:2], u_yy) - \
        np.multiply(np.multiply(ginv[:,1:2], Gamma1_22), u_x))) + u   ##change this
    if is_orig  or not is_orig:
        q = np.sum(W, axis=1)
        K =np.multiply(W*np.diag(np.array(1/q).flatten()), np.sqrt(c)*np.sqrt(c.transpose()))
        DD = np.diag(np.sum(K,axis=1))
        L= - (K-DD)/epsilon + np.eye(N)   ##change this
        uhat=np.linalg.pinv(L)@f; 
        
    else:
        model = predict_model()
        uhat = predict_u(model, c.reshape((1, -1)), y)

    # This is the error when we use the truth c, which represents the limit of our approach. 
    e1 = np.linalg.norm(u-uhat)/np.linalg.norm(u)
    print("e1: ", e1)
    
    # First, sample from the prior \pi_n
    # including the coefficients
    Lp, _, _ = GL(y,NN,1)
    eigenvalues, eigenvectors = la.eigh(Lp)
    idx = np.argsort(np.abs(eigenvalues))
    D = eigenvalues[idx][:N]
    V = eigenvectors[:, idx][:, :N]
    Lambda=(tau*np.eye(N)+D)**(-gamma/2)
    Q=V*Lambda
    Q=Q*np.sqrt(N/np.trace(Lambda**2)) #Q -- v, samples from the prior
    Q = np.loadtxt("Q_20.dat", delimiter=",")

    accept=0 
    old=np.zeros((N,1))
    kappa_old=np.exp(old) # initial value
    Nbasis=np.floor(N)
    q = np.sum(W, axis=1) # the approxiamte density
    Nbasis = int(Nbasis)
    
    if is_orig:
        K_old =np.multiply(W*np.diag(np.array(1/q).flatten()), np.sqrt(kappa_old)*np.sqrt(kappa_old.transpose()))
        DD_old = np.diag(np.sum(K_old,axis=1))
        L_old= - (K_old-DD_old)/epsilon + np.eye(N)   ##change this
        eigenvalues, eigenvectors = la.eigh(L_old.dot(L_old.T))
        idx = np.argsort(np.abs(eigenvalues))
        D_old = eigenvalues[idx][:Nbasis]
        U_old = eigenvectors[:, idx][:, :Nbasis]
        D_old = np.diag(D_old)
        D_old[0,0]=1
        D_old_inverse = np.linalg.solve(D_old, np.eye(Nbasis))
        D_old_inverse[0, 0] = 0
        p_old = L_old.T @ (U_old @ (D_old_inverse @ (U_old.T @ f)))
    else:
        ### DeepONet
        model = predict_model()
        p_old = predict_u(model, kappa_old.reshape((1, -1)), y)

    # np.random.seed(16)
    Y = u+np.random.normal(0, sigma, (N,1))
    Y_obs_set = np.array([Y[i] for i in obs_set])
    p_old_obs_set = np.array([p_old[i] for i in obs_set])
    l_old = -0.5*np.linalg.norm(Y_obs_set-p_old_obs_set)**2/(sigma**2) #the graph posterior
    pm = np.zeros((N,1))
    count = 0
    sample = np.zeros((N,1))
    W_q_diag = W * np.diag(np.array(1 / q).flatten())
    eye_Nbasis = np.eye(Nbasis)
    Q_N = Q[:,:N]
    errs = []
    accepts = []
    nums = []
    numsa = []
    t1 = time.time()
    for i in range(M):
        if i % 100 == 0:
            print(i)
        xi = np.random.normal(0,1,(N,1))
        new = np.sqrt(1-beta**2)*old+beta*Q_N@xi
        kappa_new = np.exp(new)

        if is_orig:
            """
            K_new =np.multiply(W_q_diag, np.sqrt(kappa_new)@np.sqrt(kappa_new.T))
            DD_new = np.diag(np.sum(K_new,axis=1))
            L_new= -(K_new-DD_new)/epsilon + np.eye(N) ##change this
            eigenvalues, eigenvectors = la.eigh(L_new@L_new.T)
            idx = np.argsort(np.abs(eigenvalues))
            D_new = eigenvalues[idx][:Nbasis]
            U_new = eigenvectors[:, idx][:, :Nbasis]
            D_new = np.diag(D_new)
            D_new[0,0]=1
            D_new_inverse = np.linalg.solve(D_new, eye_Nbasis)
            D_new_inverse[0, 0] = 0
            p_new = L_new.T @ (U_new @ (D_new_inverse @ (U_new.T @ f)))
            """
            K_new = scipy.sparse.csr_matrix(W_q_diag).multiply(np.sqrt(kappa_new)@np.sqrt(kappa_new.T))
            DD_new = scipy.sparse.diags(K_new.sum(axis=1).A1)
            L_new = scipy.sparse.csr_matrix(-(K_new - DD_new) / epsilon) + scipy.sparse.eye(N, format='csr')
            p_new = scipy.sparse.linalg.spsolve(L_new, f).reshape((-1,1))
            
        else:
            p_new = predict_u(model, kappa_new.reshape((1, -1)), y)

        p_new_obs_set = np.array([p_new[i] for i in obs_set])
        l_new = -0.5*np.linalg.norm(Y_obs_set-p_new_obs_set)**2/(sigma**2)

        # Overflow Handling in Exponential Calculation
        delta_l = l_new - l_old
        if delta_l > 700:
            a = 1
        else:
            a = np.exp(delta_l)
        a = min(1, a)
        b=np.random.binomial(1,a)
        if b==1: # accept with probability a
            old[:]=new
            l_old=l_new
            accept+=1
        if i+1 > 5000:
            pm+=old
            count+=1
            sample[:,count-1:count]=old
    t2 = time.time()
    print("Total and average time for each iteration: ", (t2-t1), " and ", (t2-t1)/M)
    PM=pm/count
    Var=np.sqrt(np.mean((sample-PM)**2, axis=1))
    pm_u=np.quantile(sample.transpose(),0.95)
    pm_l=np.quantile(sample.transpose(),0.05)
    e2=np.linalg.norm(np.exp(PM)-c)/np.linalg.norm(c)
    print("e2: ", e2)
    
    kappa_new = np.exp(PM)
    K_new1 = scipy.sparse.csr_matrix(W_q_diag).multiply(np.sqrt(kappa_new)@np.sqrt(kappa_new.T))
    DD_new1= scipy.sparse.diags(K_new1.sum(axis=1).A1)
    L_new1 = scipy.sparse.csr_matrix(-(K_new1 - DD_new1) / epsilon) + scipy.sparse.eye(N, format='csr')
    p_new=sla.inv(L_new1)@f; 
    p_pm = scipy.sparse.linalg.spsolve(L_new1, f).reshape((-1,1))
    e3=np.linalg.norm(u-p_pm)/np.linalg.norm(u)
    print("e3: ", e3)
    e[0,0] = e1
    e[1,0] = e2
    e[2,0] = e3

    # plt.plot(nums, errs)
    # plt.xlabel("#Iteration")
    # plt.ylabel(r"$L^2$ relative error")
    # plt.show()
    # plt.plot(numsa, accepts)
    # plt.xlabel("#Iteration")
    # plt.ylabel(r"#Accepts")
    # plt.show()
    return accept,PM,Var,pm_u,pm_l,p_pm,c,u,e


if __name__ == "__main__":
    m=20    #number of points on manifold
    NN=16    #NN = number of neighbors when constructing graph Laplacian
    sigma, beta = 0.05, 0.01  #sigma-noise level, beta-step size of pCN
    tau, gamma = 0.08, 6    #the parameter tau, s in the prior
    obs_set=[i for i in range(m**2)]     #the index set for the observation locations
    Nexp=1    #counter of experiments (because I need to run this for different paramters.) 
    M=10000   #iterations
    accept,PM,Var,pm_u,pm_l,p_pm,c,u,e=BEIP_T_DeepONet(m,NN,sigma,beta,tau,gamma,obs_set,Nexp,M, False)
    print(e)
    
    no=f"0.05_{m}_n"
    res_list = [accept,pm_u,pm_l,sigma*np.sqrt(m**2)/np.linalg.norm(u),e[0,0], e[1,0], e[2,0]]
    np.savetxt("results/res_list_20_20_{}_{}.txt".format(M,no), res_list)
    np.savetxt("results/theta_pm_sd_20_20_{}_{}.txt".format(M,no), np.hstack((PM,Var.reshape((-1, 1)))))
    np.savetxt("results/pm_u_20_20_{}_{}.txt".format(M,no), p_pm)
    np.savetxt("results/cu_20_20_{}_{}.txt".format(M,no), np.hstack((c,u)))

