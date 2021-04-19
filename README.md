# netNMF-sc
netNMF-sc: Leveraging gene-gene interactions for imputation and dimensionality reduction in single-cell expression analysis.

Preprint is available at https://www.biorxiv.org/content/10.1101/544346v1

## Instalation 
netNMF-sc is installable through pip:
pip3 install netNMFsc

Or by cloning this repository

## Running netNMF-sc
See netNMFsc_example.ipynb for a jupyter notebook tutorial for importing and running netNMF-sc. netNMF-sc can also be run from the command line using the following command:

python3 -m netNMFsc.run_netNMF-sc -x <path_to_10X_directory> --network netNMFsc/refdata/coexpedia_network_human.mtx (or a network of your choice) --netgenes netNMFsc/refdata/coexpedia_gene_names_human.npy --dimensions <integer> --max_iters <integer> --direc <directory_to_save_output> --method GD

### Command line arguments
    -x,--tenXdir, data is from 10X. Only required to provide directory containing matrix.mtx, genes.tsv, barcodes.tsv files or .hdf5 file
    -f,--filename, path to data file (.npy or .mtx)
    -g,--gene_names, path to file containing gene names (.npy or .tsv)
    -net,--network, path to network file (.npy or .mtx)
    -netgenes,--netgenes, path to file containing gene names for network (.npy or .tsv)
    -org,--organism, mouse or human
    -id,--idtype, ensemble, symbol, or entrez
    -netid,--netidtype, ensemble, symbol, or entrez
    -n,--normalize, normalize data? 1 = yes, 0 = no,default=1
    -sparse,--sparsity, sparsity for network,default=0.75
    -mi,--max_iters, max iters for netNMF,default=10000)
    -t,--tol, tolerence for netNMF,default=1e-2
    -d,--direc, directory to save files
    -D,--dimensions, number of latent dimension, default = 10
    -a,--alpha, lambda parameter from NMF,default = 10
