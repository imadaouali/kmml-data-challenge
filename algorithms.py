import numpy as np
from cvxopt import solvers, matrix
from sklearn.svm import SVC
import scipy


def KernelPCA(G, d):
    """
    Kernel PCA.
    
    G (Array): Gram Matrix
    d (int): New dimension : d << G.shape[0]
    
    X_new (Array output): Data after dimensionality reduction
    """
    n = G.shape[0]

    # Center the Gram matrix.
    U = np.ones((n, n)) / n
    G = G - U @ G - G @ U + U @ G @ U
    
    # Calculate eigenvalues and eigenvectors.
    
    eigvals, eigvecs = scipy.linalg.eigh(G)
    eigvecs = eigvecs[:, ::-1] #Sorted based on their eigenvalues (descending order)
    X_new = eigvecs[:, :d]
    
    return X_new


def sigmoid(u):
    return 1 / (1 + np.exp(-u))

def log_loss(u):
    return np.log(1 + np.exp(-u))

def distortion(lambd, n, y_train, K, alpha):
    J = 0.5 * lambd * alpha.dot(K.dot(alpha))
    for i in range(n):
        J += log_loss(y_train[i] * K[i, ].dot(alpha)) / n
    return J


def logistic_regression(K, N_split, y_train, lambd=1, precision=1e-7, max_iter=1000):
    y = 2*y_train - 1
    K_train = K[:N_split, :N_split]
    K_pred = K[N_split:, :N_split]
    n = N_split
    alpha = np.zeros(n, dtype=float)
    P = - sigmoid(-y*(K_train.dot(alpha)))
    W = np.diag(-sigmoid(y*(K_train.dot(alpha)))*P)
    z = K_train.dot(alpha) - y*P/np.diag(W)
    err = 1e5
    old_err = 0
    cur_iter = 0
    while (cur_iter < max_iter) & (np.abs(err - old_err) > precision):
        old_err = err
        W_sqrt = np.sqrt(W)

        tmp = np.linalg.inv(W_sqrt.dot(K_train).dot(W_sqrt) + lambd * n * np.eye(n))
        alpha = W_sqrt.dot(tmp).dot(W_sqrt).dot(z)
        
        m = K_train.dot(alpha)
        P = - sigmoid(-y*(m))
        W = np.diag(sigmoid(y*(m)*P))
        z = m - y*P/np.diag(W)
        
        err = distortion(lambd, n, y, K_train, alpha)
        if err - old_err > 1e-10:
            print("Distortion is going up!")
            cur_iter += 1

    return alpha, (K_pred@alpha > 0).astype('int')


def SVM_classifier(K, N_split, y_train, C=1, algo='2-SVM'):
    K_train = K[:N_split, :N_split]
    K_pred = K[N_split:, :N_split]
    if algo == 'SVM':
        P = K_train
        y = 2*y_train - 1.0
        q = -y.reshape(-1, 1)
        h = np.concatenate((C*np.ones(N_split), np.zeros(N_split)))
        G = np.concatenate((np.diag(y), -np.diag(y)))

        sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
    elif algo == '2-SVM':
        P = K_train + C*N_split*np.eye(N_split)
        y = 2*y_train - 1.0
        q = -y.reshape(-1, 1)
        G = -np.diag(y)
        h = np.zeros(N_split)
        
        sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))
    else: 
    	return 0
    alpha = np.array(sol['x'])
    
    return alpha, (K_pred@alpha > 0).astype('int')