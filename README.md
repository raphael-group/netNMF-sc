# netNMF
netNMF: A network regularization algorithm for dimensionality reduction and imputation of single-cell expression data

### Requires the following python libraries:
tensorflow or tensorflow-gpu (if running on a machine with GPUs)
numpy
scipy
anndata

## Command line arguments
    -f,--filename, path to data file (.npy or .mtx)
    -g,--gene_names, path to file containing gene names (.npy or .tsv)
    -net,--network, path to network file (.npy or .mtx)
    -netgenes,--netgenes, path to file containing gene names for network (.npy or .tsv)
    -n,--normalize, normalize data? 1 = yes, 0 = no,default=0
    -sparse,--sparsity, sparsity for network,default=0.99
    -mi,--max_iters, max iters for netNMF,default=1500)
    -t,--tol, tolerence for netNMF,default=1e-4
    -d,--direc, directory to save files
    -D,--dimensions, number of dimensions to apply shift,default = 10
    -l,--lambda_s, lambda param from NMF,default = 10
    -x,--tenXdir, data is from 10X. Only required to provide directory containing matrix.mtx, genes.tsv, barcodes.tsv files

## To run
python3 netNMF.py -x path_to_10X_directory -d /n/fs/ragr-research/projects/scRNA/data/real_data/mESCs_Buettner/coexpressdb_correlation_analysis_data/Aug26 --network network.npy --netgenes netgenes.npy --normalize 1 -l 1 -D 10

