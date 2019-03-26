from __future__ import print_function
import numpy as np
from warnings import warn
from joblib import Parallel, delayed
from . import utils
import copy,argparse,os,math,random,time
from scipy import sparse, io,linalg
from scipy.sparse import csr_matrix
from scipy.linalg import blas
import warnings
import pandas as pd
from numpy import dot,multiply
warnings.simplefilter(action='ignore', category=FutureWarning)


class netNMFGD:
    '''
    Performs netNMF-sc with gradient descent using Tensorflow
    '''
    def __init__(self, d=None, N=None, alpha=100, n_inits=1, tol=1e-2, max_iter=10000, n_jobs=1, parallel_backend='multiprocessing',normalize=True,sparsity=0.75):
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
        A = tf.constant(X.astype(np.float32))
        H =  tf.Variable(temp_H.astype(np.float32))
        W = tf.Variable(temp_W.astype(np.float32))
        WH = tf.matmul(W, H)
        L_s = tf.constant(self.L.astype(np.float32))
        alpha_s = tf.constant(np.float32(self.alpha))

        cost0 = tf.reduce_sum(tf.multiply(mask,tf.pow(A - WH, 2)))
        costL = alpha_s * tf.trace(tf.matmul(tf.transpose(W),tf.matmul(L_s,W)))

        cost = cost0 + costL

        lr = 0.002
        decay = 0.95

        global_step = tf.Variable(0, trainable=False)
        increment_global_step = tf.assign(global_step, global_step + 1)
        learning_rate = tf.train.exponential_decay(lr, global_step, self.max_iter, decay, staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1)
        train_step = optimizer.minimize(cost,global_step=global_step)

        init = tf.global_variables_initializer()
        # Clipping operation. This ensure that W and H learnt are non-negative
        clip_W = W.assign(tf.maximum(tf.zeros_like(W), W))
        clip_H = H.assign(tf.maximum(tf.zeros_like(H), H))
        clip = tf.group(clip_W, clip_H)

        c = np.inf
        with tf.Session() as sess:
            sess.run(init)
            for i in range(self.max_iter):
                sess.run(train_step)
                sess.run(clip)
                if i%300==0:
                    c2 = sess.run(cost)
                    e = c-c2
                    c = c2
                    print(i,c,e)
                    if e < self.tol:
                        conv = True
                        break
            learnt_W = sess.run(W)
            learnt_H = sess.run(H)
        tf.reset_default_graph()

        return {
            'conv': conv,
            'e': c,
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
        self.X = utils.log_transform(self.X)
        self.M = utils.get_M(self.X)
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
        best_results = {"e": np.inf, "H": None, "W": None}
        for r in results:
            if r['e'] < best_results['e']:
                best_results = r
        if 'conv' not in best_results:
            warn("Did not converge after {} iterations. Error is {}. Try increasing `max_iter`.".format(self.max_iter, best_results['e']))
        return best_results["W"], best_results["H"]


class netNMFMU:
    '''
    Performs netNMF-sc with multiplicative updates
    '''
    def __init__(self, d=None, M = None, N=None, alpha=10, n_inits=1, tol=1e-2, max_iter=1000, n_jobs=1, parallel_backend='multiprocessing',normalize=True):
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
        self.normalize=normalize

    def _init(self, X):
        # X is cells by genes
        temp_H = np.random.randn(X.shape[0], self.d).astype(np.float32)
        temp_W = np.random.randn(X.shape[1], self.d).astype(np.float32)
        temp_H = np.array(temp_H,order='F')
        temp_W = np.array(temp_W,order='F')
        return abs(temp_H),abs(temp_W)

    def _update(self, H, W, X, alpha):
        XW = dot(X,W)
        HWtW = dot(multiply(self.M,dot(H,W.T)),W)
        H = multiply(H,np.divide(XW,HWtW))

        XtH = dot(X.T,H)
        NW = alpha * dot(self.N,W)
        XtHpNW = XtH + NW
        WHtH = dot(multiply(self.M.T,dot(W,H.T)),H)
        DW = alpha * dot(self.D,W)
        WHTHpDW = WHtH + DW
        W = multiply(W,np.divide(XtHpNW,WHTHpDW))

        return H.astype(np.float32), W.astype(np.float32)


    def _error(self, H, W, X):
        return np.sum((X - multiply(self.M,dot(H,W.T)))**2) + self.alpha * np.trace(np.dot(np.dot(W.T , self.L) ,W))

    def _fit(self, X):
        X = np.multiply(self.M,X).astype(np.float32)
        H, W = self._init(X)
        print(X.shape,W.shape,H.shape)
        conv = False
        c = np.inf
        for x in range(self.max_iter):
            Hn, Wn = self._update(H, W, X, self.alpha)
            if x % 50 == 0:
                c2 = self._error(Hn,Wn,X)
                e = c-c2
                c = c2
                if e < self.tol:
                    conv = True
                    break
            H, W = Hn.astype(np.float32), Wn.astype(np.float32)
        return {
            'conv': conv,
            'e': c,
            'H': H,
            'W': W
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
        self.X = utils.log_transform(self.X).T.astype(np.float32)
        self.M = utils.get_M(self.X).astype(np.float32)
        if self.d is None:
            self.d = min(X.shape)
            print('rank set to:',self.d)
        if self.N is not None:
            if np.max(abs(self.N)) > 0:
                self.N = self.N / np.max(abs(self.N))
            N = self.N
            D = np.sum(abs(self.N),axis=0) * np.eye(self.N.shape[0])
            self.D = D
            self.N = N
            self.L = self.D - self.N
            assert utils.check_symmetric(self.L)
        else:
            self.N = np.eye(X.shape[0])
            self.D = np.eye(X.shape[0])
            self.L = self.D - self.N
        self.N = self.N.astype(np.float32)
        results = Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(delayed(self._fit)(self.X) for x in range(self.n_inits))
        best_results = {"e": np.inf, "H": None, "W": None}
        for r in results:
            if r['e'] < best_results['e']:
                best_results = r
        if 'conv' not in best_results:
            warn("Did not converge after {} iterations. Error is {}. Try increasing `max_iter`.".format(self.max_iter, best_results['e']))
        return best_results["W"], best_results["H"].T

