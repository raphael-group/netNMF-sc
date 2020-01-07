from __future__ import print_function
import numpy as np
from warnings import warn
from joblib import Parallel, delayed
from . import utils
import copy,argparse,os,math,random,time
from scipy import io,linalg
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.linalg import blas
import warnings
import pandas as pd
from numpy import dot,multiply

from math import sqrt
import warnings
import numbers
import time

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from sklearn.utils.extmath import safe_min
from sklearn.utils.validation import check_is_fitted, check_non_negative
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition.cdnmf_fast import _update_cdnmf_fast

EPSILON = np.finfo(np.float32).eps

INTEGER_TYPES = (numbers.Integral, np.integer)

class netNMFGD:
    '''
    Performs netNMF-sc with gradient descent using Tensorflow
    '''
    def __init__(self, distance="KL",d=None, N=None, alpha=10, n_inits=1, tol=1e-2, max_iter=20000, n_jobs=1, weight=0.1,parallel_backend='multiprocessing',normalize=True,sparsity=0.75,lr=0.0001):
        """
            d:          number of dimensions
            N:          Network (weighted adjacency matrix)
            alpha:      regularization parameter
            n_inits:    number of runs to make with different random inits (in order to avoid being stuck in local minima)
            n_jobs:     number of parallel jobs to run, when n_inits > 1
            tol:        stopping criteria
            max_iter:   stopping criteria
        """
        self.X = None
        self.M = None
        self.d = d
        self.N = N
        self.alpha = alpha
        self.n_inits = n_inits
        self.tol = tol
        self.max_iter = max_iter
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        self.normalize = normalize
        self.sparsity=sparsity
        self.weight = weight
        self.distance = distance
        self.lr = lr

    def _init(self, X):
        temp_H = np.random.randn(self.d,X.shape[1]).astype(np.float32)
        temp_W = np.random.randn(X.shape[0], self.d).astype(np.float32)
        temp_H = np.array(temp_H,order='F')
        temp_W = np.array(temp_W,order='F')
        return abs(temp_H),abs(temp_W)

    def _fit(self, X):
        import tensorflow as tf
        temp_H, temp_W = self._init(X)
        conv = False

        mask = tf.constant(self.M.astype(np.float32))
        eps = tf.constant(np.float32(1e-8))
        A = tf.constant(X.astype(np.float32)) + eps
        H =  tf.Variable(temp_H.astype(np.float32))
        W = tf.Variable(temp_W.astype(np.float32))
        print(np.max(mask),np.min(mask),np.sum(mask))
        WH = tf.matmul(W, H) 
        if self.weight < 1:
            WH = tf.multiply(mask,WH)
        WH += eps
        L_s = tf.constant(self.L.astype(np.float32))
        alpha_s = tf.constant(np.float32(self.alpha))
        

        if self.distance == 'frobenius':
            cost0 = tf.reduce_sum(tf.pow(A - WH, 2))
            costL = alpha_s * tf.linalg.trace(tf.matmul(tf.transpose(W),tf.matmul(L_s,W)))
        elif self.distance == 'KL':
            cost0 = tf.reduce_sum(tf.multiply(A ,tf.math.log(tf.div(A,WH)))-A+WH)
            costL = alpha_s * tf.trace(tf.matmul(tf.transpose(W),tf.matmul(L_s,W)))
        else:
            raise ValueError('Select frobenius or KL for distance')

        if self.alpha > 0:
            cost = cost0 + costL
        else:
            cost = cost0

        lr = self.lr
        decay = 0.95

        global_step = tf.Variable(0, trainable=False)
        increment_global_step = tf.compat.v1.assign(global_step, global_step + 1)
        learning_rate = tf.compat.v1.train.exponential_decay(lr, global_step, self.max_iter, decay, staircase=True)

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate, epsilon=.1)
        train_step = optimizer.minimize(cost,global_step=global_step)

        init = tf.compat.v1.global_variables_initializer()
        # Clipping operation. This ensure that W and H learnt are non-negative
        clip_W = tf.compat.v1.assign(W,tf.maximum(tf.zeros_like(W), W))
        clip_H = tf.compat.v1.assign(H,tf.maximum(tf.zeros_like(H), H))
        clip = tf.group(clip_W, clip_H)

        c = np.inf
        with tf.compat.v1.Session() as sess:
            sess.run(init)
            for i in range(self.max_iter):
                sess.run(train_step)
                sess.run(clip)
                if i%300==0:
                    c2 = sess.run(cost)
                    e = c-c2
                    c = c2
                    if i%10000==0:
                        print(i,c,e)
                    if e < self.tol:
                        conv = True
                        break
            learnt_W = sess.run(W)
            learnt_H = sess.run(H)
        tf.reset_default_graph()

        return {
            'conv': conv,
            'obj': c,
            'H': learnt_H,
            'W': learnt_W
        }

    def load_10X(self,direc=None,genome='mm10'):
        if direc.endswith('hdf5') or direc.endswith('h5'):
            X,genenames = utils.import_10X_hdf5(direc,genome)
        else:
            X,genenames = utils.import_10X_mtx(direc)
        self.X = X
        self.genes = genenames

    def load_network(self,net=None,genenames=None,sparsity=.75):
        if net:
            if net.endswith('.txt'):
                network,netgenes = utils.import_network_from_gene_pairs(net,genenames)
            else:
                network,netgenes = utils.import_network(net,genenames,sparsity)
        network = utils.network_threshold(network,sparsity)
        self.N = network
        self.netgenes = netgenes


    def fit_transform(self, X=None):
        if type(X) == np.ndarray:
            self.X = X
        if type(self.genes) == np.ndarray and type(self.netgenes) == np.ndarray: # if imported data from file reorder network to match genes in X
            assert type(self.X) == np.ndarray
            assert type(self.N) == np.ndarray
            network = utils.reorder(self.genes,self.netgenes,self.N,self.sparsity)
            self.N = network
            self.netgenes = self.genes
        if self.normalize:
            print('library size normalizing...')
            self.X = utils.normalize(self.X)
        #self.X = utils.log_transform(self.X)
        M = np.ones_like(self.X)
        M[self.X == 0] = self.weight
        self.M = M
        if self.d is None:
            self.d = min(X.shape)
            print('rank set to:',self.d)
        if self.N is not None:
            if np.max(abs(self.N)) > 0:
                self.N = self.N / np.max(abs(self.N))
            N = self.N
            D = np.sum(abs(self.N),axis=0) * np.eye(self.N.shape[0])
            print(np.count_nonzero(N),'edges')
            self.D = D
            self.N = N
            self.L = self.D - self.N
            assert utils.check_symmetric(self.L)
        else:
            self.N = np.eye(X.shape[0])
            self.D = np.eye(X.shape[0])
            self.L = self.D - self.N
        
        results = Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(delayed(self._fit)(self.X) for x in range(self.n_inits))
        best_results = {"obj": np.inf, "H": None, "W": None}
        for r in results:
            if r['obj'] < best_results['obj']:
                best_results = r
        if 'conv' not in best_results:
            warn("Did not converge after {} iterations. Error is {}. Try increasing `max_iter`.".format(self.max_iter, best_results['e']))
        return(best_results['W'],best_results['H'])



# NMF code is adapted from from Non-negative matrix factorization in scikit=learn library


def norm(x):
    """Dot product-based Euclidean norm implementation

    See: http://fseoane.net/blog/2011/computing-the-vector-norm/

    Parameters
    ----------
    x : array-like
        Vector for which to compute the norm
    """
    return sqrt(squared_norm(x))


def trace_dot(X, Y):
    """Trace of np.dot(X, Y.T).

    Parameters
    ----------
    X : array-like
        First matrix
    Y : array-like
        Second matrix
    """
    return np.dot(X.ravel(), Y.ravel())


def _check_init(A, shape, whom):
    A = check_array(A)
    if np.shape(A) != shape:
        raise ValueError('Array with wrong shape passed to %s. Expected %s, '
                         'but got %s ' % (whom, shape, np.shape(A)))
    check_non_negative(A, whom)
    if np.max(A) == 0:
        raise ValueError('Array passed to %s is full of zeros.' % whom)


def _beta_divergence(lam,N,D,X, W, H, beta, square_root=False):
    L = D - N

    beta = _beta_loss_to_float(beta)

    # The method can be called with scalars
    if not sp.issparse(X):
        X = np.atleast_2d(X)
    W = np.atleast_2d(W)
    H = np.atleast_2d(H)

    # Frobenius norm
    if beta == 2:
        # Avoid the creation of the dense np.dot(W, H) if X is sparse.
        if sp.issparse(X):
            norm_X = np.dot(X.data, X.data)
            norm_WH = trace_dot(np.dot(np.dot(W.T, W), H), H)
            cross_prod = trace_dot((X * H.T), W)
            res = (norm_X + norm_WH - 2. * cross_prod) / 2.
        else:
            res = squared_norm(X - np.dot(W, H)) / 2.

        if square_root:
            return np.sqrt(res * 2)
        else:
            return res

    if sp.issparse(X):
        # compute np.dot(W, H) only where X is nonzero
        WH_data = _special_sparse_dot(W, H, X).data
        X_data = X.data
    else:
        WH = np.dot(W, H)
        WH_data = WH.ravel()
        X_data = X.ravel()

    # do not affect the zeros: here 0 ** (-1) = 0 and not infinity
    indices = X_data > EPSILON
    WH_data = WH_data[indices]
    X_data = X_data[indices]

    # used to avoid division by zero
    WH_data[WH_data == 0] = EPSILON

    # generalized KL divergence
    if beta == 1:
        # fast and memory efficient computation of np.sum(np.dot(W, H))
        sum_WH = np.dot(np.sum(W, axis=0), np.sum(H, axis=1))
        # computes np.sum(X * log(X / WH)) only where X is nonzero
        div = X_data / WH_data
        res = np.dot(X_data, np.log(div))
        # add full np.sum(np.dot(W, H)) - np.sum(X)
        res += sum_WH - X_data.sum()
        res += lam * np.trace(np.dot(np.dot(W.T,L),W)) ### netNMF

    # Itakura-Saito divergence
    elif beta == 0:
        div = X_data / WH_data
        res = np.sum(div) - np.product(X.shape) - np.sum(np.log(div))

    # beta-divergence, beta not in (0, 1, 2)
    else:
        if sp.issparse(X):
            # slow loop, but memory efficient computation of :
            # np.sum(np.dot(W, H) ** beta)
            sum_WH_beta = 0
            for i in range(X.shape[1]):
                sum_WH_beta += np.sum(np.dot(W, H[:, i]) ** beta)

        else:
            sum_WH_beta = np.sum(WH ** beta)

        sum_X_WH = np.dot(X_data, WH_data ** (beta - 1))
        res = (X_data ** beta).sum() - beta * sum_X_WH
        res += sum_WH_beta * (beta - 1)
        res /= beta * (beta - 1)

    if square_root:
        res = np.sqrt(2 * res)
        print(res)
        return res
    else:
        print(res)
        return res


def _special_sparse_dot(W, H, X):
    """Computes np.dot(W, H), only where X is non zero."""
    if sp.issparse(X):
        ii, jj = X.nonzero()
        dot_vals = np.multiply(W[ii, :], H.T[jj, :]).sum(axis=1)
        WH = sp.coo_matrix((dot_vals, (ii, jj)), shape=X.shape)
        return WH.tocsr()
    else:
        return np.dot(W, H)


def _compute_regularization(alpha, l1_ratio, regularization):
    """Compute L1 and L2 regularization coefficients for W and H"""
    alpha_H = 0.
    alpha_W = 0.
    if regularization in ('both', 'components'):
        alpha_H = float(alpha)
    if regularization in ('both', 'transformation'):
        alpha_W = float(alpha)

    l1_reg_W = alpha_W * l1_ratio
    l1_reg_H = alpha_H * l1_ratio
    l2_reg_W = alpha_W * (1. - l1_ratio)
    l2_reg_H = alpha_H * (1. - l1_ratio)
    return l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H


def _check_string_param(solver, regularization, beta_loss, init):

    beta_loss = _beta_loss_to_float(beta_loss)
    return beta_loss


def _beta_loss_to_float(beta_loss):
    """Convert string beta_loss to float"""
    allowed_beta_loss = {'frobenius': 2,
                         'KL': 1,
                         'itakura-saito': 0}
    if isinstance(beta_loss, str) and beta_loss in allowed_beta_loss:
        beta_loss = allowed_beta_loss[beta_loss]

    if not isinstance(beta_loss, numbers.Number):
        raise ValueError('Invalid beta_loss parameter: got %r instead '
                         'of one of %r, or a float.' %
                         (beta_loss, allowed_beta_loss.keys()))
    return beta_loss


def _initialize_nmf(X, n_components, init=None, eps=1e-6,
                    random_state=None):

    check_non_negative(X, "NMF initialization")
    n_samples, n_features = X.shape

    if (init is not None and init != 'random'
            and n_components > min(n_samples, n_features)):
        raise ValueError("init = '{}' can only be used when "
                         "n_components <= min(n_samples, n_features)"
                         .format(init))

    if init is None:
        if n_components <= min(n_samples, n_features):
            init = 'nndsvd'
        else:
            init = 'random'

    # Random initialization
    if init == 'random':
        avg = np.sqrt(X.mean() / n_components)
        rng = check_random_state(random_state)
        H = avg * rng.randn(n_components, n_features)
        W = avg * rng.randn(n_samples, n_components)
        # we do not write np.abs(H, out=H) to stay compatible with
        # numpy 1.5 and earlier where the 'out' keyword is not
        # supported as a kwarg on ufuncs
        np.abs(H, H)
        np.abs(W, W)
        return W, H

    # NNDSVD initialization
    U, S, V = randomized_svd(X, n_components, random_state=random_state)
    W, H = np.zeros(U.shape), np.zeros(V.shape)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if init == "nndsvd":
        pass
    elif init == "nndsvda":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif init == "nndsvdar":
        rng = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * rng.randn(len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * rng.randn(len(H[H == 0])) / 100)
    else:
        raise ValueError(
            'Invalid init parameter: got %r instead of one of %r' %
            (init, (None, 'random', 'nndsvd', 'nndsvda', 'nndsvdar')))

    return W, H




def _multiplicative_update_w(lam,N,D,X, W, H, beta_loss, l1_reg_W, l2_reg_W, gamma,
                             H_sum=None, HHt=None, XHt=None, update_H=True):
    """update W in Multiplicative Update NMF"""
    if beta_loss == 2:
        # Numerator
        if XHt is None:
            XHt = safe_sparse_dot(X, H.T)
        if update_H:
            # avoid a copy of XHt, which will be re-computed (update_H=True)
            numerator = XHt
        else:
            # preserve the XHt, which is not re-computed (update_H=False)
            numerator = XHt.copy()

        # Denominator
        if HHt is None:
            HHt = np.dot(H, H.T)
        denominator = np.dot(W, HHt)

    else:
        # Numerator
        # if X is sparse, compute WH only where X is non zero
        WH_safe_X = _special_sparse_dot(W, H, X)
        if sp.issparse(X):
            WH_safe_X_data = WH_safe_X.data
            X_data = X.data
        else:
            WH_safe_X_data = WH_safe_X
            X_data = X
            # copy used in the Denominator
            WH = WH_safe_X.copy()
            if beta_loss - 1. < 0:
                WH[WH == 0] = EPSILON

        # to avoid taking a negative power of zero
        if beta_loss - 2. < 0:
            WH_safe_X_data[WH_safe_X_data == 0] = EPSILON

        if beta_loss == 1:
            np.divide(X_data, WH_safe_X_data, out=WH_safe_X_data)
            C = np.dot(W,W.T)
            numerator = safe_sparse_dot(WH_safe_X_data , H.T ) + lam * np.dot(N,W)
            
        
        elif beta_loss == 0:
            # speeds up computation time
            # refer to /numpy/numpy/issues/9363
            WH_safe_X_data **= -1
            WH_safe_X_data **= 2
            # element-wise multiplication
            WH_safe_X_data *= X_data
            numerator = safe_sparse_dot(WH_safe_X, H.T)
        else:
            WH_safe_X_data **= beta_loss - 2
            # element-wise multiplication
            WH_safe_X_data *= X_data
            numerator = safe_sparse_dot(WH_safe_X, H.T)



        # Denominator
        if beta_loss == 1:
            if H_sum is None:
                H_sum = np.sum(H, axis=1)  # shape(n_components, )
            denominator = H_sum[np.newaxis, :] + lam * np.dot(D,W)

        else:
            # computation of WHHt = dot(dot(W, H) ** beta_loss - 1, H.T)
            if sp.issparse(X):
                # memory efficient computation
                # (compute row by row, avoiding the dense matrix WH)
                WHHt = np.empty(W.shape)
                for i in range(X.shape[0]):
                    WHi = np.dot(W[i, :], H)
                    if beta_loss - 1 < 0:
                        WHi[WHi == 0] = EPSILON
                    WHi **= beta_loss - 1
                    WHHt[i, :] = np.dot(WHi, H.T)
            else:
                WH **= beta_loss - 1
                WHHt = np.dot(WH, H.T)
            denominator = WHHt

    # Add L1 and L2 regularization
    if l1_reg_W > 0:
        denominator += l1_reg_W
    if l2_reg_W > 0:
        denominator = denominator + l2_reg_W * W
    denominator[denominator == 0] = EPSILON

    numerator /= denominator
    delta_W = numerator

    # gamma is in ]0, 1]
    if gamma != 1:
        delta_W **= gamma

    return delta_W, H_sum, HHt, XHt


def _multiplicative_update_h(X, W, H, beta_loss, l1_reg_H, l2_reg_H, gamma):
    """update H in Multiplicative Update NMF"""
    if beta_loss == 2:
        numerator = safe_sparse_dot(W.T, X)
        denominator = np.dot(np.dot(W.T, W), H)

    else:
        # Numerator
        WH_safe_X = _special_sparse_dot(W, H, X)
        if sp.issparse(X):
            WH_safe_X_data = WH_safe_X.data
            X_data = X.data
        else:
            WH_safe_X_data = WH_safe_X
            X_data = X
            # copy used in the Denominator
            WH = WH_safe_X.copy()
            if beta_loss - 1. < 0:
                WH[WH == 0] = EPSILON

        # to avoid division by zero
        if beta_loss - 2. < 0:
            WH_safe_X_data[WH_safe_X_data == 0] = EPSILON

        if beta_loss == 1:
            np.divide(X_data, WH_safe_X_data, out=WH_safe_X_data)
        elif beta_loss == 0:
            # speeds up computation time
            # refer to /numpy/numpy/issues/9363
            WH_safe_X_data **= -1
            WH_safe_X_data **= 2
            # element-wise multiplication
            WH_safe_X_data *= X_data
        else:
            WH_safe_X_data **= beta_loss - 2
            # element-wise multiplication
            WH_safe_X_data *= X_data

        # here numerator = dot(W.T, (dot(W, H) ** (beta_loss - 2)) * X)
        numerator = safe_sparse_dot(W.T, WH_safe_X)

        # Denominator
        if beta_loss == 1:
            W_sum = np.sum(W, axis=0)  # shape(n_components, )
            W_sum[W_sum == 0] = 1.
            denominator = W_sum[:, np.newaxis]

        # beta_loss not in (1, 2)
        else:
            # computation of WtWH = dot(W.T, dot(W, H) ** beta_loss - 1)
            if sp.issparse(X):
                # memory efficient computation
                # (compute column by column, avoiding the dense matrix WH)
                WtWH = np.empty(H.shape)
                for i in range(X.shape[1]):
                    WHi = np.dot(W, H[:, i])
                    if beta_loss - 1 < 0:
                        WHi[WHi == 0] = EPSILON
                    WHi **= beta_loss - 1
                    WtWH[:, i] = np.dot(W.T, WHi)
            else:
                WH **= beta_loss - 1
                WtWH = np.dot(W.T, WH)
            denominator = WtWH

    # Add L1 and L2 regularization
    if l1_reg_H > 0:
        denominator += l1_reg_H
    if l2_reg_H > 0:
        denominator = denominator + l2_reg_H * H
    denominator[denominator == 0] = EPSILON

    numerator /= denominator
    delta_H = numerator

    # gamma is in ]0, 1]
    if gamma != 1:
        delta_H **= gamma

    return delta_H


def _fit_multiplicative_update(lam,N,D,X, W, H, beta_loss='frobenius',
                               max_iter=200, tol=1e-4,
                               l1_reg_W=0, l1_reg_H=0, l2_reg_W=0, l2_reg_H=0,
                               update_H=True, verbose=0):

    start_time = time.time()

    beta_loss = _beta_loss_to_float(beta_loss)

    # gamma for Maximization-Minimization (MM) algorithm [Fevotte 2011]
    if beta_loss < 1:
        gamma = 1. / (2. - beta_loss)
    elif beta_loss > 2:
        gamma = 1. / (beta_loss - 1.)
    else:
        gamma = 1.

    # used for the convergence criterion
    error_at_init = _beta_divergence(lam,N,D,X, W, H, beta_loss, square_root=True)
    previous_error = error_at_init

    H_sum, HHt, XHt = None, None, None
    for n_iter in range(1, max_iter + 1):
        # update W
        # H_sum, HHt and XHt are saved and reused if not update_H
        delta_W, H_sum, HHt, XHt = _multiplicative_update_w(lam,N,D,
            X, W, H, beta_loss, l1_reg_W, l2_reg_W, gamma,
            H_sum, HHt, XHt, update_H)
        W *= delta_W

        # necessary for stability with beta_loss < 1
        if beta_loss < 1:
            W[W < np.finfo(np.float64).eps] = 0.

        # update H
        if update_H:
            delta_H = _multiplicative_update_h(X, W, H, beta_loss, l1_reg_H,
                                               l2_reg_H, gamma)
            H *= delta_H

            # These values will be recomputed since H changed
            H_sum, HHt, XHt = None, None, None

            # necessary for stability with beta_loss < 1
            if beta_loss <= 1:
                H[H < np.finfo(np.float64).eps] = 0.

        # test convergence criterion every 10 iterations
        if tol > 0 and n_iter % 10 == 0:
            error = _beta_divergence(lam,N,D,X, W, H, beta_loss, square_root=True)

            if verbose:
                iter_time = time.time()
                print("Epoch %02d reached after %.3f seconds, error: %f" %
                      (n_iter, iter_time - start_time, error))

            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    # do not print if we have already printed in the convergence test
    if verbose and (tol == 0 or n_iter % 10 != 0):
        end_time = time.time()
        print("Epoch %02d reached after %.3f seconds." %
              (n_iter, end_time - start_time))

    return W, H, n_iter


def non_negative_factorization(lam,N,D,X, W=None, H=None, n_components=None,
                               init='warn', update_H=True, solver='mu',
                               beta_loss='KL', tol=1e-4,
                               max_iter=400, alpha=0., l1_ratio=0.,
                               regularization=None, random_state=None,
                               verbose=0, shuffle=False):

    X = check_array(X, accept_sparse=('csr', 'csc'), dtype=float)
    check_non_negative(X, "NMF (input X)")
    beta_loss = _check_string_param(solver, regularization, beta_loss, init)

    if safe_min(X) == 0 and beta_loss <= 0:
        raise ValueError("When beta_loss <= 0 and X contains zeros, "
                         "the solver may diverge. Please add small values to "
                         "X, or use a positive beta_loss.")

    n_samples, n_features = X.shape
    if n_components is None:
        n_components = n_features

    if not isinstance(n_components, INTEGER_TYPES) or n_components <= 0:
        raise ValueError("Number of components must be a positive integer;"
                         " got (n_components=%r)" % n_components)
    if not isinstance(max_iter, INTEGER_TYPES) or max_iter < 0:
        raise ValueError("Maximum number of iterations must be a positive "
                         "integer; got (max_iter=%r)" % max_iter)
    if not isinstance(tol, numbers.Number) or tol < 0:
        raise ValueError("Tolerance for stopping criteria must be "
                         "positive; got (tol=%r)" % tol)

    if init == "warn":
        if n_components < n_features:
            warnings.warn("The default value of init will change from "
                          "random to None in 0.23 to make it consistent "
                          "with decomposition.NMF.", FutureWarning)
        init = "random"

    # check W and H, or initialize them
    if init == 'custom' and update_H:
        _check_init(H, (n_components, n_features), "NMF (input H)")
        _check_init(W, (n_samples, n_components), "NMF (input W)")
    elif not update_H:
        _check_init(H, (n_components, n_features), "NMF (input H)")

        avg = np.sqrt(X.mean() / n_components)
        W = np.full((n_samples, n_components), avg)

    else:
        W, H = _initialize_nmf(X, n_components, init=init,
                               random_state=random_state)

    l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H = _compute_regularization(
        alpha, l1_ratio, regularization)

    W, H, n_iter = _fit_multiplicative_update(lam,N,D,X, W, H, beta_loss, max_iter,
                                                  tol, l1_reg_W, l1_reg_H,
                                                  l2_reg_W, l2_reg_H, update_H,
                                                  verbose)


    if n_iter == max_iter and tol > 0:
        warnings.warn("Maximum number of iteration %d reached. Increase it to"
                      " improve convergence." % max_iter, ConvergenceWarning)

    return W, H, n_iter


class netNMFMU(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=None, init=None, solver='mu',
                 beta_loss='KL', tol=1e-4, max_iter=400,
                 random_state=None, alpha=0., l1_ratio=0., verbose=0,
                 shuffle=False):
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.verbose = verbose
        self.shuffle = shuffle

    def fit_transform(self, lam,N,X, y=None, W=None, H=None):
        """Learn a NMF model for the data X and returns the transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be decomposed

        y : Ignored

        W : array-like, shape (n_samples, n_components)
            If init='custom', it is used as initial guess for the solution.

        H : array-like, shape (n_components, n_features)
            If init='custom', it is used as initial guess for the solution.

        Returns
        -------
        W : array, shape (n_samples, n_components)
            Transformed data.
        """
        D = np.sum(abs(N),axis=0) * np.eye(N.shape[0])
        N = np.dot(np.linalg.inv(D),N)
        print(N.shape)
        D = np.eye(D.shape[0])
        print(D.shape)
        X = check_array(X, accept_sparse=('csr', 'csc'), dtype=float)

        W, H, n_iter_ = non_negative_factorization(lam=lam,N=N,D=D,
            X=X, W=W, H=H, n_components=self.n_components, init=self.init,
            update_H=True, solver=self.solver, beta_loss=self.beta_loss,
            tol=self.tol, max_iter=self.max_iter, alpha=self.alpha,
            l1_ratio=self.l1_ratio, regularization='both',
            random_state=self.random_state, verbose=self.verbose,
            shuffle=self.shuffle)

        self.reconstruction_err_ = _beta_divergence(lam,N,D,X, W, H, self.beta_loss,
                                                    square_root=True)

        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_iter_ = n_iter_

        return W

    def fit(self, lam,N,D,X, y=None, **params):
        """Learn a NMF model for the data X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be decomposed

        y : Ignored

        Returns
        -------
        self
        """
        self.fit_transform(lam,N,D,X, **params)
        return self

    def transform(self, lam,N,D,X):
        """Transform the data X according to the fitted NMF model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be transformed by the model

        Returns
        -------
        W : array, shape (n_samples, n_components)
            Transformed data
        """
        check_is_fitted(self, 'n_components_')

        W, _, n_iter_ = non_negative_factorization(
            lam=lam,N=N,D=D,X=X, W=None, H=self.components_, n_components=self.n_components_,
            init=self.init, update_H=False, solver=self.solver,
            beta_loss=self.beta_loss, tol=self.tol, max_iter=self.max_iter,
            alpha=self.alpha, l1_ratio=self.l1_ratio, regularization='both',
            random_state=self.random_state, verbose=self.verbose,
            shuffle=self.shuffle)

        return W

    def inverse_transform(self, W):
        """Transform data back to its original space.

        Parameters
        ----------
        W : {array-like, sparse matrix}, shape (n_samples, n_components)
            Transformed data matrix

        Returns
        -------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix of original shape

        .. versionadded:: 0.18
        """
        check_is_fitted(self, 'n_components_')
        return np.dot(W, self.components_)

