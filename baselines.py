"""
"""
import numpy as np
from timeit import default_timer as timer
import time, re, sys, os
import pandas as pd
import random
from scipy.sparse import coo_matrix, csc_matrix, save_npz, load_npz
import csv

from aux import utils
from aux.gen_settings import gen_graph_dag_tril, gen_graph_dag, \
                             gen_data_sem_original, \
                              gen_data_sem_nv, \
                            gen_bipartite_latent2observed, \
                            convert_clustering2adjmatrix, \
                            simulate_data_linearLVM, \
                            simulate_data_linearLVM_NV \

import networkx as nx
from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import KMeans

#---R packages
from rpy2 import robjects
from rpy2.robjects import numpy2ri

def CLV1(Xobs, K=3, sX=False, sXr=False, verbose=False):
    """
    Reference: https://cran.r-project.org/web/packages/ClustVarLV/ClustVarLV.pdf
    Call this solver by
            robjects.r('library("ClustVarLV")')

            _ = robjects.r.assign("X", numpy2ri.py2rpy(Xobs))
            robjects.r(' .. ')

            # --Scenario tutorial:
            data(apples_sh)
            #directional groups
            resclvX <- CLV(X = apples_sh$senso, method = "directional", sX = TRUE)
            resclvkmX <- CLV_kmeans(X = apples_sh$senso, method = "directional", clust = 3, sX = FALSE, sXr = FALSE)

            # --- Scenario 2: test with data saved from numpy array
            # install.packages("reticulate")
            library(reticulate)
            # Load NumPy
            np <- import("numpy")
            # Read the .npy file
            data <- np$load("your_file.npy")
            #directional groups
            resclvX <- CLV(X = data, method = "directional", sX = TRUE)
            resclvkmX <- CLV_kmeans(X = data, method = "directional", clust = K, sX = FALSE, sXr = FALSE)

            # Output: Get results using
            robjects.r('labels <- get_partition(resclv, K = NULL, type = "vector")')
            labels = numpy2ri.rpy2py(robjects.r('as(labels, "vector")'))
    """
    if sX == False:
        sX_ = 'FALSE'
    else:
        sX_ = 'TRUE'
    if sXr == False:
        sXr_ = 'FALSE'
    else:
        sXr_ = 'TRUE'
    robjects.r('library("ClustVarLV")')
    _ = robjects.r.assign("data", numpy2ri.py2rpy(Xobs))
    #
    robjects.r('resclv <- CLV(X = data, method = "directional", sX = %s)'%sX_)
    #
    robjects.r('labels <- get_partition(resclv, K = %d, type = "vector")' %K)
    labels = numpy2ri.rpy2py(robjects.r('as(labels, "vector")'))
    return labels

def CLV2(Xobs, K=3, sX=False, sXr=False, verbose=False):
    """
    Reference: https://cran.r-project.org/web/packages/ClustVarLV/ClustVarLV.pdf
    Call this solver by
            robjects.r('library("ClustVarLV")')

            _ = robjects.r.assign("X", numpy2ri.py2rpy(Xobs))
            robjects.r(' .. ')

            # --Scenario tutorial:
            data(apples_sh)
            #directional groups
            resclvX <- CLV(X = apples_sh$senso, method = "directional", sX = TRUE)
            resclvkmX <- CLV_kmeans(X = apples_sh$senso, method = "directional", clust = 3, sX = FALSE, sXr = FALSE)

            # --- Scenario 2: test with data saved from numpy array
            # install.packages("reticulate")
            library(reticulate)
            # Load NumPy
            np <- import("numpy")
            # Read the .npy file
            data <- np$load("your_file.npy")
            #directional groups
            resclvX <- CLV(X = data, method = "directional", sX = TRUE)
            resclvkmX <- CLV_kmeans(X = data, method = "directional", clust = K, sX = FALSE, sXr = FALSE)

            # Output: Get results using
            robjects.r('labels <- get_partition(resclv, K = NULL, type = "vector")')
            labels = numpy2ri.rpy2py(robjects.r('as(labels, "vector")'))
    """
    if sX == False:
        sX_ = 'FALSE'
    else:
        sX_ = 'TRUE'
    if sXr == False:
        sXr_ = 'FALSE'
    else:
        sXr_ = 'TRUE'
    robjects.r('library("ClustVarLV")')
    _ = robjects.r.assign("data", numpy2ri.py2rpy(Xobs))
    #
    robjects.r('resclv <- CLV_kmeans(X = data, method = "directional", clust = %d, sX = %s, sXr = %s)'%(K, sX_, sXr_))

    robjects.r('labels <- get_partition(resclv, K = %d, type = "vector")' %K)
    labels = numpy2ri.rpy2py(robjects.r('as(labels, "vector")'))
    return labels



#---FOFC
def run_fofc(X ):
    # Load dataset
    df = pd.DataFrame( X )
    df.columns = df.columns.astype(str)

    t0 = timer()
    # # Start Java VM
    from pycausal.pycausal import pycausal as pc
    pc = pc()
    pc.start_vm()

    # Load causal algorithms from the py-causal library and Run FOFC Continuous
    from pycausal import search as s
    tetrad = s.tetradrunner()
    tetrad.getAlgorithmParameters(algoId = 'fofc')

    tetrad.run(algoId = 'fofc', dfs = df, alpha = 0.01, useWishart = False, useGap = False, include_structure_model = False, verbose = True)
    #----
    t1 = timer() - t0

    # Assuming tetrad_graph is the result of tetrad.getTetradGraph()
    tetrad_graph = tetrad.getTetradGraph()

    # Step 1: Get the nodes and edges
    nodes = tetrad_graph.getNodes()  # List of nodes
    #
    edges = tetrad.getEdges()  # List of edges
    # Step 2: Create a mapping from node name to index
    node_to_index = {str(node): idx for idx, node in enumerate(nodes)}

    # Step 3: Initialize an adjacency matrix
    n = len(nodes)
    adj_matrix = np.zeros((n, n))
    labels = np.array([0 for _ in range(d)])
    # Step 4: Record results
    for edge_string in edges:
        print(edge_string)  # Output: "_L1 --> Sympathy"
        # Step 1: Split the string around the arrow
        if "-->" in edge_string:
            parts = edge_string.split("-->")
            # Step 2: Strip whitespace and surrounding quotes
            nodes_ = [part.strip().strip("'\'") for part in parts]
            #
            source = nodes_[0]
            target = nodes_[1]
            #
            s_id = int(source[2:])
            t_id = int( target )
            labels[t_id] = s_id
        elif "<--" in edge_string:
            parts = edge_string.split("<--")
            nodes_ = [part.strip().strip("'\'") for part in parts]
            source = nodes_[1]
            target = nodes_[0]
        else:
            print("The edge string format is unknown!")
        #----
        # Get the indices of the source and target nodes
        source_idx = node_to_index[source]
        target_idx = node_to_index[target]
        adj_matrix[source_idx, target_idx] = 1
    print(edges)
    print(adj_matrix)
    pc.stop_vm()
    return labels


def _input_args(args=None):
    if args==None:
        args = {}
    for a in sys.argv:
        if "=" in a:
            p_k, p_v = a.split("=")
            p_v = p_v.split(",")
            if p_k in ['ds', 'k']:
                args[p_k] = [int(_) for _ in p_v]
            elif p_k in ['degs','var_varNs', 'max_scaling', 'std_interp','rad_err', 'ice_lambda1', 'opts']:
                args[p_k] = [float(_) for _ in p_v]
            elif p_k in ['rnd', 'LAM', 'TAUX_RHO', 'ALM_LS_MAX','EPSILON', 'ALM_LS_BETA']:
                args[p_k] = float(p_v[0])
            elif p_k in ['SEED','ipara','MAXITER_ALM','MAXITER_PRIMAL','toplot','VAR', 'verbo']:
                args[p_k] = int(p_v[0])
            elif p_k in ['graph_type','sem_type','fdir', 'fout', 'filename', 'dataset', \
                            'solver_primal', 'rowname']:
                args[p_k] = p_v[0]
            elif p_k == "runwho" or p_k == "alg":
                #
                args[p_k] = [str(_) for _ in p_v]
            elif p_k == "baseline":
                args[p_k] = [str(_) for _ in p_v]
            else:
                print("Unknown parameter:", p_k)
                pass
    return args


if __name__ == '__main__':
    timestr = time.strftime("%H%M%S%m%d")
    args={'ds': [50], \
         'degs': [1.0, 2.0, 3.0], \
        'graph_type': 'ER' , 'sem_type': 'gauss', 'SEED':0,\
        'rnd': 10, \
        'runwho': [], 'baseline': 'na', 'toplot': 0, 'filename':'', \
        'LAM': 0.1, 'opts': [0.1], \
        'ipara': 0, 'fdir':'', 'fout':'', \
        'VAR': 0, 'verbo': 2, \
        }
    args = _input_args(args)
    filename = args['filename']

    if args['runwho'][0] in ['syn-ev', 'syn-oracle', 'syn-ice-emp']:
        """Example
        """
        fdir = args['fdir']
        fout = args['fout']
        verbo = args['verbo']

        graph_type = args['graph_type']
        sem_type = args['sem_type']
        SEED = args['SEED']
        d = args['ds'][0]
        k = args['k'][0]
        if d < 3*k:
            print('The number of observed variables should be larger than 3 x the number of latent variables !')
            d = 5*k
        deg = args['degs'][0]
        rnd = args['rnd']
        opts = args['opts']
        fout = args['fout']
        dir_out = '%s/%s'%(fdir,fout)
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)

        # Generate random graph and observational data
        nn = int(np.ceil(rnd * d))
        X, Cw_true, Bw_true, Pi_true, labels_true, Y_latent, X_pre, weights_mode \
                = simulate_data_linearLVM(d, nn, k, opts, SEED, deg, \
                                         graph_type=graph_type, sem_type=sem_type)
    elif args['runwho'][0] in ['syn-nv', 'syn-hetero']:
        """Example
        """
        fdir = args['fdir']
        fout = args['fout']
        verbo = args['verbo']

        graph_type = args['graph_type']
        sem_type = args['sem_type']
        SEED = args['SEED']
        d = args['ds'][0]
        k = args['k'][0]
        if d < 3*k:
            print('The number of observed variables should be larger than 3 x the number of latent variables !')
            d = 5*k
        deg = args['degs'][0]
        rnd = args['rnd']
        opts = args['opts'] # an array of numbers, default form: 0.1,0.3,10.0
        fout = args['fout']
        dir_out = '%s/%s'%(fdir,fout)
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)

        # Generate random graph and observational data
        nn = int(np.ceil(rnd * d))
        X, Cw_true, Bw_true, Pi_true, labels_true, Y_latent, X_pre, weights_mode \
                = simulate_data_linearLVM_NV(d, nn, k, opts, SEED, deg, \
                                         graph_type=graph_type, sem_type=sem_type)
    else:
        print('Data generation type not known.')
    # Save to files
    if verbo > 1:
        np.save('%s/datamat.npy'%dir_out, X)
    save_npz('%s/Cw_true.npz'%(dir_out), csc_matrix(Cw_true) )
    save_npz('%s/Bw_true.npz'%(dir_out), csc_matrix(Bw_true) )

    print('Generated synthetic data X of size: ', X.shape)
    print('...the data matrix X(:10, :15) looks like: ', X[:10,:15])

    baseline=args['baseline'][0]
    print('\n\nBaseline is: ', baseline)

    if baseline in ['CLV', 'ClustVarLV']:
        t0 = timer()
        #--- Import CLV
        labels = CLV2( X, K=k )
        t1 = timer() - t0
    elif baseline in ['fofc', 'FOFC']:
        t0 = timer()
        #---
        labels = run_fofc( X )
        t1 = timer() - t0

    # Evaluate clustering score
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    # Adjusted Rand Index (ARI)
    ari = adjusted_rand_score(labels_true, labels)
    print("\n(%s) Adjusted Rand Index:"%baseline, ari)
    # Normalized Mutual Information (NMI)
    nmi = normalized_mutual_info_score(labels_true, labels)
    print("(%s) Normalized Mutual Information:"%baseline, nmi)

    # --- Evaluate results
    noise_var = opts[2]
    runwho = args['runwho'][0]
    print('\n\ngraph_type,deg,sem_type,mode_edgeWeights,evnv,noise_var,d,k,rnd,seed,alg,ARI,NMI,time,time_total')
    print("%s,%d,%s,%s,%s,%.2f,%d,%d,%.3f,%d,%s,%.3f,%.3f,%.3f,%.3f"%(graph_type, deg, sem_type, weights_mode, runwho, noise_var, \
            d,k, rnd, SEED, baseline, ari, nmi, t1, t1))


