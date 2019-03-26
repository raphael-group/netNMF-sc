# utils for netNMF-sc
from __future__ import print_function
import numpy as np
from warnings import warn
import copy,os,math,random,time
from scipy import sparse, io
from scipy.sparse import csr_matrix
import warnings
import pandas as pd
import mygene
from scipy.sparse import csgraph
warnings.simplefilter(action='ignore', category=FutureWarning)



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
def network_threshold(net,sparsity):
    net = net / np.max(abs(net))
    # m = np.max(abs(net))
    # for i in range(net.shape[0]):
    #     net[i,i] = m
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
def laplacian(net):
    D = np.sum(abs(net),axis=0) * np.eye(net.shape[0])
    Dpow = np.sqrt(D)
    Dpow = Dpow + 1e-10
    Dpow = 1 / Dpow
    Dpow[Dpow >= 1e10] = 0
    N = np.dot(np.dot(Dpow,net),Dpow)
    return np.eye(net.shape[0]),N

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
data_genes: names of genes in the data
network_genes:  names of genes in the network
network: gene x gene numpy array 

@output
new_network: gene x gene numpy array 
'''
def get_gene(x):
    if 'symbol' in x.keys():
        return x['symbol']
    return ''

def reorder(data_genes,network_genes,network,sparsity): 
    data_genes = [gene.lower() for gene in data_genes]
    network_genes = [gene.lower() for gene in network_genes]

    data_idtype = get_geneid_type(data_genes)
    network_idtype = get_geneid_type(network_genes)

    mg = mygene.MyGeneInfo()
    if data_idtype != 'symbol':
    	geneSyms_X = [get_gene(x).lower() for x in mg.querymany(data_genes , scopes=["ensemblgene","entrezgene"], fields='symbol')]
    else:
    	geneSyms_X = data_genes
    if network_idtype != 'symbol':
        geneSyms_net =[get_gene(x).lower() for x in mg.querymany(network_genes , scopes=["ensemblgene","entrezgene"], fields='symbol')]
    else:
    	geneSyms_net = network_genes

    inds = []
    zeroinds = []
    for i,gene in enumerate(geneSyms_X):
    	if gene in geneSyms_net:
    		inds.append(geneSyms_net.index(gene))
    	else:
    		inds.append(0)
    		zeroinds.append(i)

    new_network = network[inds,:]
    new_network = new_network[:,inds]
    new_network[zeroinds,:] = 0
    new_network[:,zeroinds] = 0
    new_network = network_threshold(new_network,sparsity)
    assert check_symmetric(new_network)
    return new_network

'''
@input
genes: n x 1 matrix of gene ids

@output
geneid: ensemble, entrez, or symbol
'''
def get_geneid_type(genes):
    if genes[0][0:3] == 'ENS' or genes[0][0:3] == 'ens':
        return 'ensemble'
    if RepresentsInt(genes[0]): # entrez ids
        return 'entrez'
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
    if os.path.isfile(os.path.join(directory,'genes.tsv')):
        genes = pd.read_csv(os.path.join(directory,'genes.tsv'), header=None, sep='\t')
    elif os.path.isfile(os.path.join(directory,'features.tsv')):
        genes = pd.read_csv(os.path.join(directory,'features.tsv'), header=None, sep='\t')
    else:
        raise Exception('No genes.tsv or features.tsv file present')
    genes = np.asarray(genes)
    genes = np.asarray([x[-1] for x in genes])
    return X,genes

'''
@input
fname: path to directory containing matrix.mtx, genes.tsv, and barcodes.tsv
# TODO: redo using h5py 
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
        except tables.NoSuchNodeError:
            raise Exception('Genome %s does not exist in this file.' % genome)
        except KeyError:
            raise Exception('File is missing one or more required datasets.')
    genes = dsets['genes'].astype(str)
    genes = np.asarray(genes)
    genes = np.asarray([x[-1] for x in genes])
    return matrix.toarray(),genes

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
    return X,genes

'''
@input
network_file: path to network in csv format
genename_file: path to genenames in csv format

@output
network
network_genes
'''
def import_network_csv(network_file,genename_file,sep='\t',sparsity=.95):
    network = pd.read_csv(network_file,header=None,sep=sep)
    network_genes = pd.read_csv(genename_file,header=None,sep=sep)
    return network,np.asarray(network_genes)

'''
@input
network_file: path to network in csv format
genename_file: path to genenames in csv format

@output
network
network_genes
'''
def import_network_mtx(network_file,genename_file,sparsity):
    network = load_mtx(network_file)
    names = load_mtx(genename_file)
    return network,np.asarray(network_genes)

def import_npy(filename,gene_names,network_file,genename_file,sparsity):
    X = np.load(filename)
    genes = np.load(gene_names)
    network,network_genes = import_network_npy(network_file,genename_file,sparsity)
    return X,genes,network,network_genes
'''
@input
a: Anndata object
network_file: path to network in npy format
genename_file: path to genenames in npy format

@output
network
network_genes
'''
def import_network(network_file,genename_file,sparsity):
    s = time.time()
    if network_file.endswith('npy'):
        network = load_npy(network_file)
    elif network_file.endswith('mtx'):
        network = load_mtx(network_file)
    else:
        raise Exception('network must be in .npy or .mtx format')
    if genename_file.endswith('npy'):
        network_genes = load_npy(genename_file)
    elif genename_file.endswith('mtx'):
        network_genes = load_mtx(genename_file)
    else:
        raise Exception('network gene names must be in .npy or .mtx format')
    return network,network_genes

'''
@input
filename: path to network in in gene pairs format (gene1\tgene2\tedge weight\n)

@output
net
'''
def import_network_from_gene_pairs(filename):
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
    return net

'''
@input
X

@output
X
'''
def normalize(X):
    X = X.astype(np.float32)
    library_sizes = []
    for i in range(X.shape[1]):
        library_size = float(sum(X[:,i]))
        library_sizes.append(library_size)
        X[:,i] = X[:,i]/float(library_size)
    X = X * np.median(library_sizes)
    return X

def get_M(X):
    M = np.ones_like(X)
    M[X == 0] = .1
    return M

def log_transform(X):
    if np.max(X) >= 15:
        print('log-tansforming X with pseudocount 1')
        return np.log(X+1)
    return X




