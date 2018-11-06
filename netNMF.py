from __future__ import print_function
import tensorflow as tf
import numpy as np
from scipy.io import mminfo,mmread
from scipy import sparse, io
import sys,itertools,collections
import copy,argparse,os,math,random,time
from scipy import sparse, io
import pandas as pd
from anndata import AnnData, \
    read_csv, read_excel, read_text, read_hdf, read_mtx
from netNMFmu import NMF
import tables
from scipy.sparse import csr_matrix
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn import decomposition
from sklearn.neighbors import kneighbors_graph
import sklearn.preprocessing
import sklearn.decomposition


'''
@input
X: matrix; a n(genes) by m(cells) matrix with gene counts
d: scalar; latent factor dimensions
lambada_s: scalar; regularization parameter
L: matrix; is n by n covariance matrix known from another study

@output
learnt_W, n x d
learnt_H, d x m
'''
def run_TF(a,steps):
    d = a.uns['d']
    lambda_s = a.uns['lambda_s']
    net = a.uns['network']
    weight = a.uns['weight']
    X = a.X.T

    lambda_s = float(lambda_s)
    L = laplacian(net).astype(np.float32)

    X = X.astype(np.float32)
    assert X.shape[0] == L.shape[0] and X.shape[0] == L.shape[1]
    assert d < X.shape[1]
    shape = X.shape
    #number of latent factors
    rank = d

    mask = tf.Variable(weight)

    A = tf.constant(X)
    # Initializing random H and W
    temp_H = np.random.randn(rank, shape[1]).astype(np.float32)
    temp_H = np.divide(temp_H, temp_H.max())

    temp_W = np.random.randn(shape[0], rank).astype(np.float32)
    temp_W = np.divide(temp_W, temp_W.max())

    H =  tf.Variable(temp_H)
    W = tf.Variable(temp_W)
    WH = tf.matmul(W, H)

    L_s = tf.constant(L)
    alpha_s = tf.constant(lambda_s)

    cost0 = tf.reduce_sum(tf.pow(tf.boolean_mask(A, mask) - tf.boolean_mask(WH, mask), 2))

    costL = alpha_s * (tf.trace(tf.matmul(tf.transpose(W),tf.matmul(L_s,W))))

    if lambda_s > 0:
        print('running netNMF...')
        cost = cost0 + costL
    else:
        print('running NMF...')
        cost = cost0

    lr = 0.05
    decay = 0.95

    global_step = tf.Variable(0, trainable=False)
    increment_global_step = tf.assign(global_step, global_step + 1)
    learning_rate = tf.train.exponential_decay(lr, global_step, steps, decay, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=.0001)
    #optimizer = tf.train.AdagradOptimizer(learning_rate)
    train_step = optimizer.minimize(cost,global_step=global_step)

    init = tf.global_variables_initializer()

    # Clipping operation. This ensure that W and H learnt are non-negative
    clip_W = W.assign(tf.maximum(tf.zeros_like(W), W))
    clip_H = H.assign(tf.maximum(tf.zeros_like(H), H))
    clip = tf.group(clip_W, clip_H)
    s = time.time()
    #with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    with tf.Session() as sess:
        sess.run(init)
        for i in range(steps):
            sess.run(train_step)
            sess.run(clip)
            if i%200==0:
                print("Cost: %f" % sess.run(cost))
        learnt_W = sess.run(W)
        learnt_H = sess.run(H)
    tf.reset_default_graph()
    return learnt_W, learnt_H

''' 
@input
filepath: path to .mtx file

@output
X: numpy array
'''
def load_mtx(filepath):
    X = io.mmread(filepath).toarray()
    return X

''' 
@input
filepath: path to .npy file

@output
X: numpy array
'''
def load_npy(filepath):
    X = np.load(filepath)
    return X

'''
@input
net: gene x gene numpy array with entries corresponding to edge weights

@output
net: gene x gene numpy array thresholded at predetermined weight
'''
def network_threshold(net,sparsity=.95):
    net = net / np.max(abs(net))
    m = np.max(abs(net))
    for i in range(net.shape[0]):
        net[i,i] = m
    s = abs(net.flatten())
    init = (len(s)-np.count_nonzero(net))/float(len(s))
    diff = sparsity - init
    num = int(diff * len(s))
    if num > 0:
        s = s[s > 0]
        s.sort()
        threshold = s[num]
        net[abs(net) < threshold] = 0
    nonzero = np.count_nonzero(net)
    print(nonzero,'edges in network')
    return net


'''
@input
L: matrix

@output
Bool: True if symmetric
'''
def check_symmetric(L, tol=1e-8):
    return np.allclose(L, L.T, atol=tol)

'''
@input
L: matrix

@output
Bool: is positive semidefinite with small error
'''
def is_pos_sdef(L):
    e = np.linalg.eigvals(L)
    return np.all(e >= -1e-5)

'''
@input 
net: gene x gene numpy array representing a network

@output:
L: laplacian of network
'''
from scipy.sparse import csgraph
def laplacian(net):
    normalized = True
    assert(check_symmetric(net))
    d = np.sum(abs(net),axis=0)
    I = np.eye(net.shape[0])
    L=I*d-net
    if normalized:
        osd = np.zeros(len(d))
        for i in range(len(d)):
            if d[i]>0:
                osd[i] = np.sqrt(1.0/d[i])
        T = I*osd
        L = np.dot(T,np.dot(L,T))

    assert(check_symmetric(L))
    assert(is_pos_sdef(L))

    return L

'''
@input
data: matrix

@output
Bool: True if data is count data (rather than log-normalized)
'''
def iscount(data):
    print('max value in data',np.max(data))
    return np.max(data) > 20.0

'''
@input
s: string

@output
Bool: True if string represents an integer
'''
def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

'''
@input
organism: string (human or mouse)

@output
genedict: Dictionary containing conversion between entrez, ensemble, and gene symbols
'''
def get_conversion_info(organism):
    if organism == 'human':
        fh = 'gene_id_conversion_table_human.csv'
        with open(fh) as f:
            genes = f.read().split('\n')[0:-1]
            # ensemble, gene name, entrez gene number
            genes = [x.split(',') for x in genes]
    elif organism == 'mouse':
        fh = 'gene_id_conversion_table_mouse.csv'
        with open(fh) as f:
            genes = f.read().split('\n')[0:-1]
            # ensemble, gene name, entrez gene number
            genes = [x.split(',') for x in genes]
    else:
        raise Exception('Organism must be human or mouse')
    genedict = dict()
    genedict['entrez'] = []
    genedict['ensemble'] = []
    genedict['symbol'] = []
    for gene in genes:
        (n,ensemble,symbol,entrez) = gene
        genedict['entrez'].append(entrez)
        genedict['ensemble'].append(ensemble.lower())
        genedict['symbol'].append(symbol.lower())
    return genedict

'''
@input
a: Anndata
network_genes: n x 1 array containing names of genes in the network
network: n x n Matrix 

@output
a: Anndata object containing network with columns/rows reordered to match ordering of data a.X
'''
def reorder(a,network_genes,network): 
    if network.shape[0] == a.X.T.shape[0]:
        a.uns['network'] = network
        return a
    data_genes = a.var['gene_ids'].values
    data_genes = [gene.lower() for gene in data_genes]
    network_genes = [gene.lower() for gene in network_genes]

    data_idtype = get_geneid_type(data_genes)
    network_idtype = get_geneid_type(network_genes)
    organism = get_organism(data_genes,data_idtype)

    
    conversion_info = get_conversion_info(organism)
    network = np.hstack((network,np.zeros([network.shape[0],1])))
    network = np.vstack((network,np.zeros([1,network.shape[1]])))
    new_network = network.copy()
    mapping = data_genes.copy()
    inds = []
    for i,gene in enumerate(data_genes):
        try:
            if data_idtype == network_idtype:
                formatted_gene = gene
            else:
                formatted_gene = conversion_info[network_idtype][conversion_info[data_idtype].index(gene)]
            network_location = network_genes.index(formatted_gene)
            inds.append(network_location)
        except:
            inds.append(len(network_genes))
    new_network = new_network[inds,:]
    new_network = new_network[:,inds]

    a.uns['network'] = new_network

    return a

'''
@input
genes: n x 1 matrix of gene ids
idtype: gene id type (entrez, ensemble, symbol)

@output
organism: 'human' or 'mouse'
'''
def get_organism(genes,idtype):
    if idtype == 'ensemble':
        if genes[0][3] == 'G':
            return 'human'
        elif genes[0][3:6] == 'MUS':
            return 'mouse'
        else:
            raise Exception('Not human or mouse data')
    human_genes = get_conversion_info('human')
    mouse_genes = get_conversion_info('mouse')
    if idtype == 'entrez': # entrez ids
        overlaps_human = len(set(genes).intersection(set(human_genes[idtype])))
        overlaps_mouse = len(set(genes).intersection(set(mouse_genes[idtype])))
        if overlaps_human > overlaps_mouse:
            return 'human'
        else:
            return 'mouse'
    else:
        genes = np.char.lower(genes)
        overlaps_human = len(set(genes).intersection(set(human_genes[idtype])))
        overlaps_mouse = len(set(genes).intersection(set(mouse_genes[idtype])))
        if overlaps_human > overlaps_mouse:
            return 'human'
        else:
            return 'mouse'

'''
@input
genes: n x 1 matrix of gene ids

@output
geneid: ensemble, entrez, or symbol
'''
def get_geneid_type(genes):
    if genes[0][0:3] == 'ENS':
        return 'ensemble'
    if RepresentsInt(genes[0]): # entrez ids
        return 'entrez'
    else:
        return 'symbol'

'''
@input
directory: path to directory containing matrix.mtx, genes.tsv, and barcodes.tsv

@output
a: Anndata object
'''
def import_10X_mtx(directory):
    start = time.time()
    X = load_mtx(os.path.join(directory,'matrix.mtx'))
    genes = pd.read_csv(os.path.join(directory,'genes.tsv'), header=None, sep='\t')
    if len(genes) == X.shape[0]: # transpose if matrix is genes x cells
        a = AnnData(X.T)
    else:
        a.X = Anndata(X)
    var_names = genes[1]
    a.var_names = var_names
    a.var['gene_ids'] = genes[0].values
    a.obs_names = pd.read_csv(os.path.join(directory,'barcodes.tsv'), header=None)[0]
    a.uns['network'] = np.ones([a.X.shape[1],a.X.shape[1]])
    return a

'''
@input
fname: path to directory containing matrix.mtx, genes.tsv, and barcodes.tsv

@output
a: Anndata object
'''
def import_10X_hdf5(fname,genome='mm10'):
    with tables.open_file(str(fname), 'r') as f:
        try:
            dsets = {}
            for node in f.walk_nodes('/' + genome, 'Array'):
                dsets[node.name] = node.read()
            M, N = dsets['shape']
            data = dsets['data']
            if dsets['data'].dtype == np.dtype('int32'):
                data = dsets['data'].view('float32')
                data[:] = dsets['data']
            matrix = csr_matrix((data, dsets['indices'], dsets['indptr']),
                                shape=(N, M))
            a = AnnData(matrix.toarray())
            a.obs_names = dsets['barcodes'].astype(str)
            a.var_names = dsets['gene_names'].astype(str)
            a.var['gene_ids'] = dsets['genes'].astype(str)
        except tables.NoSuchNodeError:
            raise Exception('Genome %s does not exist in this file.' % genome)
        except KeyError:
            raise Exception('File is missing one or more required datasets.')
    a.uns['network'] = np.ones([a.X.shape[1],a.X.shape[1]])
    return a

'''
@input
data_file: path to matrix in csv format
gene_file: path to gene names in csv format
cell_file: path to cell ids in csv format

@output
a: Anndata object
'''
def import_csv(data_file,gene_file,cell_file='',sep='\t'):
    X = pd.read_csv(data_file,header=None,sep=sep)
    genes = pd.read_csv(os.path.join(path,'genes.tsv'), header=None, sep=sep)
    if len(genes) == X.shape[0]: # transpose if matrix is genes x cells
        a = AnnData(X.T)
    else:
        a = Anndata(X)
    var_names = genes[1]
    a.var_names = var_names
    a.var['gene_ids'] = genes[0].values
    if cell_file:
        a.obs_names = pd.read_csv(cell_file, header=None,sep=sep)[0]
    a.uns['network'] = np.ones([a.X.shape[1],a.X.shape[1]])
    return a

'''
@input
a: Anndata object
network_file: path to network in csv format
genename_file: path to genenames in csv format

@output
a: Anndata object
'''
def import_network_csv(a,network_file,genename_file,sep='\t',sparsity=.99):
    network = pd.read_csv(network_file,header=None,sep=sep)
    network = network_threshold(network,sparsity=sparsity)
    names = pd.read_csv(genename_file,header=None,sep=sep)
    a = reorder(a,names,network)
    return a

'''
@input
a: Anndata object
network_file: path to network in mtx format
genename_file: path to genenames in mtx format

@output
a: Anndata object
'''
def import_network_mtx(a,network_file,genename_file,sparsity=.99):
    network = load_mtx(network_file)
    network = network_threshold(network,sparsity=sparsity)
    names = load_mtx(genename_file)
    a = reorder(a,names,network)
    return a

'''
@input
a: Anndata object
network_file: path to network in npy format
genename_file: path to genenames in npy format

@output
a: Anndata object
'''
def import_network_npy(a,network_file,genename_file,sparsity):
    s = time.time()
    network = load_npy(network_file)
    network = network_threshold(network,sparsity=sparsity)
    names = load_npy(genename_file)
    a = reorder(a,names,network)
    print('finished importing network',time.time()-s)
    return a

'''
@input
a: Anndata object
filename: path to network in in gene pairs format (gene1\tgene2\tedge weight\n)

@output
a: Anndata object
'''
def import_network_from_gene_pairs(a,filename):
    with open(filename) as f:
        genepairs = f.read().split('\n')
    genepairs = [genes.split('\t') for genes in genepairs]
    genes = list(set([gene for genes in genepairs for gene in genes if not gene.replace('.','',1).isdigit()]))
    genes = [gene for gene in genes if len(gene) > 2]
    network = np.zeros([len(genes),len(genes)])
    for g in genepairs:
        if len(g) > 1:
            gene1 = g[0]
            gene2 = g[1]
            if len(g)>2 and g[2].replace('.','',1).isdigit():
                num = float(g[2])
            else:
                num = 1
            try:
                ind1 = genes.index(gene1)
                ind2 = genes.index(gene2)
                network[ind1,ind2] = num
                network[ind2,ind1] = num
            except:
                continue
    net = reorder(a,genes,network)
    return net

'''
@input
a: Anndata object

@output
a: Anndata object, where a.X is normalized
'''
def normalize(a):
    X = a.X.T
    X = X.astype(np.float32)
    library_sizes = []
    for i in range(X.shape[1]):
        library_size = float(sum(X[:,i]))
        library_sizes.append(library_size)
        X[:,i] = X[:,i]/float(library_size)
    X = X * np.median(library_sizes)
    a.X = X.T
    return a

'''
@input
a: Anndata object
weight: n x m weight matrix
dimensions: number of latend dimensions
lambda_s: regularization parameter
max_iters: maximum number of iterations for gradient descent
tol: tolerance for gradient descent

@output
a: Anndata object
'''
def run_netNMF(a,weight=[],dimensions=10,lambda_s = 1.0,max_iters=1000,tol=1e-4):
    if iscount(a.X):
        print('data is being log transformed with pseudocount 1')
        a.X = np.log(a.X+1)
    a.uns['d'] = dimensions
    a.uns['lambda_s'] = lambda_s
    if len(weight) == 0:
        weight = a.X.T > 0
    a.uns['weight'] = weight
    ### run netNMF ###
    learnt_W,learnt_H = run_TF(a,max_iters)
    ### save result in anndata object ###
    a.uns['W'] = learnt_W
    a.uns['H'] = learnt_H
    imputed = np.exp(np.dot(learnt_W,learnt_H))-1
    a.X = imputed.T
    return a

'''
@input
a: Anndata object
filepath: path to write file
'''
def save_output(a,filepath):
    a.write(filepath)

'''
@input
a: Anndata object

@output
weight: n x m weight matrix
'''
def get_weight(a):
    X = a.X.T
    median_expressions = [np.median(X[gene,X[gene]>0]) for gene in range(X.shape[0])]
    avg_expression = np.median(X[X>0])
    weight = np.zeros_like(X)
    weight[X != 0] = 1
    for gene in range(X.shape[0]):
        weight[gene,X[gene]==0] = weight[gene,X[gene]==0] + min(1,median_expressions[gene]/avg_expression)
    return weight

def main(args):
    if args.tenXdir:
        a = import_10X_mtx(args.tenXdir)
    if args.lambda_s:
        if args.network.endswith('.txt'):
            a = import_network_from_gene_pairs(a,args.network)
        else:
            a = import_network_npy(a,args.network,args.netgenes,args.sparsity)
        net = a.uns['network']
        a.uns['network'] = net 
    if args.normalize:
        a = normalize(a)
    a = run_netNMF(a,[],args.dimensions,args.lambda_s,args.max_iters,args.tol)
    print('Saving data to %s'%os.path.join(args.direc,'netNMF.h5ad'))
    save_output(a,os.path.join(args.direc,'netNMF.h5ad'))
    return a


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--filename", help="path to data file (.npy or .mtx)",type=str,default='matrix.mtx')
    parser.add_argument("-g","--gene_names", help="path to file containing gene names (.npy or .tsv)",type=str,default='gene_names.tsv')
    parser.add_argument("-net","--network", help="path to network file (.npy or .mtx)",type=str,default='')
    parser.add_argument("-netgenes","--netgenes", help="path to file containing gene names for network (.npy or .tsv)",type=str,default='')
    parser.add_argument("-org","--organism", help="mouse or human",type=str,default='human')
    parser.add_argument("-id","--idtype", help="ensemble, symbol, or entrez",type=str,default='ensemble')
    parser.add_argument("-netid","--netidtype", help="ensemble, symbol, or entrez",type=str,default='entrez')
    parser.add_argument("-n","--normalize", help="normalize data? 1 = yes, 0 = no",type=int,default=0)
    parser.add_argument("-sparse","--sparsity", help="sparsity for network",type=float,default=0.99)
    parser.add_argument("-mi","--max_iters", help="max iters for netNMF",type=int,default=1000)
    parser.add_argument("-t","--tol", help="tolerence for netNMF",type=float,default=1e-4)
    parser.add_argument("-d","--direc", help="directory to save files",default='')
    parser.add_argument("-D","--dimensions", help="number of dimensions to apply shift",type=int,default = 10)
    parser.add_argument("-l","--lambda_s", help="lambda param from NMF",type=float,default = 1.0)
    parser.add_argument("-m","--min_count", help="minimum number of transcripts per gene",type=float,default = 0.0)
    parser.add_argument("-x","--tenXdir", help="data is from 10X. Only required to provide directory containing matrix.mtx, genes.tsv, barcodes.tsv files",type=str,default = '')
    args = parser.parse_args()
    main(args)

