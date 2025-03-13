"""
# LVlearn: learning measurement models via subspace identification and clustering

Contact: shuyu.dong@centralesupelec.fr
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
# Evaluate clustering score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from baselines import CLV2, CLV1

def gendata_sem_fromDAG(B_true, n=200, sem_type='gauss', seed=1):
    utils.set_random_seed(seed)
    W_true = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(W_true, n, sem_type)
    return W_true, X

def _get_undir_graph(Ci):
    # Encode edges (nonzeros) by -1 for
    # utils.count_accuracy to recognize undirected graphs
    C = Ci.copy()
    C -= np.diag(np.diag(C))
    C[C!=0] = -1
    return np.tril(C)

def HeteroPCA(X, hatSigma=None, rank=5, max_iter=10):
    n_samples, d = X.shape
    if hatSigma == None:
        Xc = X - np.mean(X, axis=0, keepdims=True)
        hatSigma = np.dot(Xc.T, Xc) / n_samples
    # Init
    M = hatSigma - np.diag(np.diag(hatSigma))
    df = [0]
    for niter in range(max_iter):
        M_old = M.copy()
        (U, S, _) = np.linalg.svd( M )
        Uk = U[:,:rank]
        Sk = S[:rank]
        tM = (Uk @ np.diag(Sk)) @ Uk.T
        M -= np.diag(np.diag(M))
        M += np.diag( np.diag(tM) )
        df.append( np.linalg.norm( M - M_old ) )
    return (Uk, Sk, df)

def clustering_std_flipping(tU, Sigma_lr=None, eps=0.01):
    if Sigma_lr == None:
        Sigma_lr = tU @ tU.T

    d, rk = tU.shape[0], tU.shape[1]
    norm2_rows = np.diag(Sigma_lr)
    D_std = np.diag(1/np.sqrt(norm2_rows))
    Sigmalr_std = ( D_std @ Sigma_lr ) @ D_std
    Scopy = Sigmalr_std.copy()

    # Selective Flipping
    tU_new = D_std @ tU
    for i in range(d):
        for j in range(i+1,d):
            if Scopy[i, j] < -(1-eps) :
                tU_new[j,:] *= -1
                Scopy[:, j] *= -1 # necessary adjustment
                Scopy[j, :] *= -1 # necessary adjustment
    kmeans = KMeans(n_clusters=rk, random_state=0).fit(tU_new)
    return kmeans.labels_

def lvlearn_heteroskedastic(X, hatSigma=None, rank=5, max_iter=10, eps=0.01):
    # HeteroPCA + selectiveFlip
    n_samples, d = X.shape
    if hatSigma is None:
        Xc = X - np.mean(X, axis=0, keepdims=True)
        hatSigma = np.dot(Xc.T, Xc) / n_samples
    # Init
    M = hatSigma - np.diag(np.diag(hatSigma))
    df = [0]
    for niter in range(max_iter):
        M_old = M.copy()
        (U, S, _) = np.linalg.svd( M )
        Uk = U[:,:rank]
        Sk = S[:rank]
        tM = (Uk @ np.diag(Sk)) @ Uk.T
        M -= np.diag(np.diag(M))
        M += np.diag( np.diag(tM) )
        df.append( np.linalg.norm( M - M_old ) )
    # return (Uk, Sk, df)
    # ---- selective flipping then k-means
    tU = Uk @ np.diag(np.sqrt(Sk))
    Sigma_lr = tU @ tU.T
    # d, rk = tU.shape[0], tU.shape[1]
    norm2_rows = np.diag(Sigma_lr)
    D_std = np.diag(1/np.sqrt(norm2_rows))
    Sigmalr_std = ( D_std @ Sigma_lr ) @ D_std
    Scopy = Sigmalr_std.copy()

    # Selective Flipping for k-means
    tU_new = D_std @ tU
    for i in range(d):
        for j in range(i+1,d):
            if Scopy[i, j] < -(1-eps) :
                tU_new[j,:] *= -1
                Scopy[:, j] *= -1 #
                Scopy[j, :] *= -1 #
    kmeans = KMeans(n_clusters=rank, random_state=0).fit(tU_new)
    return kmeans.labels_, df, tU_new

def lvlearn_homoskedastic(X, hatSigma=None, rank=5, max_iter=10, eps=0.01):
    # k-PCA + selectiveFlip
    n_samples, d = X.shape
    if hatSigma == None:
        Xc = X - np.mean(X, axis=0, keepdims=True)
        hatSigma = np.dot(Xc.T, Xc) / n_samples
    # Init
    (U,S,Ut) = np.linalg.svd( hatSigma )
    smin = S[-1]
    Uk = U[:,:rank]
    Sk = S[:rank] - smin
    # return (Uk, Sk)

    # ---- Selective flipping then k-means
    tU = Uk @ np.diag(np.sqrt(Sk))
    Sigma_lr = tU @ tU.T
    #
    norm2_rows = np.diag(Sigma_lr)
    D_std = np.diag(1/np.sqrt(norm2_rows))
    Sigmalr_std = ( D_std @ Sigma_lr ) @ D_std
    Scopy = Sigmalr_std.copy()

    # Selective Flipping
    tU_new = D_std @ tU
    for i in range(d):
        for j in range(i+1,d):
            if Scopy[i, j] < -(1-eps) :
                tU_new[j,:] *= -1
                Scopy[:, j] *= -1 #
                Scopy[j, :] *= -1 #
    kmeans = KMeans(n_clusters=rank, random_state=0).fit(tU_new)
    return kmeans.labels_, tU_new


def lvlearn_ev2(X, SVD=None, hatSigma=None, rank=5, max_iter=10, eps=0.01):
    # k-PCA + selectiveFlip
    n_samples, d = X.shape
    if SVD is None:
        if hatSigma is None:
            Xc = X - np.mean(X, axis=0, keepdims=True)
            hatSigma = np.dot(Xc.T, Xc) / n_samples
        (U,S,Ut) = np.linalg.svd( hatSigma )
    else:
        U, S = SVD[0], SVD[1]
    smin = S[-1]
    Uk = U[:,:rank]
    Sk = S[:rank] - smin

    # ---- Selective flipping then k-means
    tU = Uk @ np.diag(np.sqrt(Sk))
    Sigma_lr = tU @ tU.T
    #
    norm2_rows = np.diag(Sigma_lr)
    D_std = np.diag(1/np.sqrt(norm2_rows))
    Sigmalr_std = ( D_std @ Sigma_lr ) @ D_std
    Scopy = Sigmalr_std.copy()

    # Selective Flipping
    tU_new = D_std @ tU
    for i in range(d):
        for j in range(i+1,d):
            if Scopy[i, j] < -(1-eps) :
                tU_new[j,:] *= -1
                Scopy[:, j] *= -1 #
                Scopy[j, :] *= -1 #
    kmeans = KMeans(n_clusters=rank, random_state=0).fit(tU_new)
    return kmeans.labels_, tU_new

"""
    Model selection functions
"""
def _topK_by_ratio(S, ratio=0.9):
    total_sum = S.sum()
    cumsum_S = np.cumsum(S)
    k = np.searchsorted(cumsum_S, ratio * total_sum) + 1  # +1 for 1-based indexing
    return k
def _convert_labels2centroids(Y, label_func ):
    # initialize C as a np array of the same size as Y (the mxk PC matrix)
    C = np.zeros( Y.shape )
    label_names = np.unique(label_func)
    for label in label_names:
        #
        centroid = np.mean(Y[label_func == label, :], axis = 0)
        C[label_func==label, :] = centroid
    return C
# The SON objective function
def _SON_objective_function(C, Y, gamma):
    m, k = C.shape
    l2_part = 0.5 * ((Y-C)**2).sum()
    Delta = np.array( [ (C[i,:] - C[j,:]) for i in range(m-1) for j in range(i+1,m) ] )
    penalty_l1 = abs(Delta).sum()
    return l2_part + gamma*penalty_l1
def _estimate_opt( fvals ):
    # Find the key (minimizer) with the minimum value
    min_key = min(fvals, key=fvals.get)
    # Get the minimum value
    min_value = fvals[min_key]
    return min_key, min_value
def _get_tail3( fvals ):
    mini3_indices = sorted(fvals, key=fvals.get)[:3]
    return mini3_indices

def lvlearn_modelSel_EV(X, hatSigma=None, gamma=0.2, eps=0.01, labels_true=None):
    n_samples, d = X.shape
    if hatSigma == None:
        Xc = X - np.mean(X, axis=0, keepdims=True)
        hatSigma = np.dot(Xc.T, Xc) / n_samples
    # Init
    (U,S,Ut) = np.linalg.svd( hatSigma )
    kmin = max(5, _topK_by_ratio(S, 0.7))
    kmax = min(d-2, max(19, _topK_by_ratio(S, 0.98)))
    #
    kopt_new = 0
    fvalues = {}
    runtimes = {}
    nmis = {}
    fopts = {}
    labels_opt = np.array([])
    Y_opt = []
    list_labels = {}
    list_Y = {}
    for kk in range(kmin,kmax+1):
        t0 = timer()
        labels, Y = lvlearn_ev2(X, SVD=(U,S), rank=kk, eps=eps )
        tt = timer() - t0
        list_labels[kk] = labels
        list_Y[kk] = Y
        runtimes[kk] = tt
        C = _convert_labels2centroids(Y, labels)
        fvalues[kk] =  _SON_objective_function(C, Y, gamma=gamma)
        # Normalized Mutual Information (NMI)
        if labels_true is not None:
            nmis[kk] = normalized_mutual_info_score(labels_true, labels)
        else:
            nmis[kk] = np.nan
        # if both median(kopt, kopt2, kopt3) is smaller than kk-3
        # then adopt the integer upper(median)
        mini3_indices = _get_tail3( fvalues )
        kstar_ = np.median( mini3_indices )
        if (kstar_ < kk - 3) :
            if not kstar_.is_integer():
                kstar = int(kstar_) + 1
            else:
                kstar = int(kstar_)
            # Exception: when min( fvalues ) is really small
            if fvalues[mini3_indices[0]] < 0.1:
                kstar = mini3_indices[0]
            labels_opt = list_labels[kstar]
            Y_opt = list_Y[kstar]
            fopts[kstar] = fvalues[kstar]
            break
        # ##-----
    if len(labels_opt) < 1:
        print('debugging model sel function: ')
        print('kmin,kmax:', kmin, kmax)
        print(fvalues)
    return (labels_opt, Y_opt, runtimes, nmis, fopts, fvalues)

def lvlearn_modelSel_NV(X, hatSigma=None, gamma=0.2, eps=0.01, labels_true=None):
    n_samples, d = X.shape
    if hatSigma is None:
        Xc = X - np.mean(X, axis=0, keepdims=True)
        hatSigma = np.dot(Xc.T, Xc) / n_samples
    # Init
    (U,S,Ut) = np.linalg.svd( hatSigma )
    kmin = max(5, _topK_by_ratio(S, 0.7))
    kmax = min(d-2, max(19, _topK_by_ratio(S, 0.98)))
    ## --debug
    kmin = 5
    kmax = 16
    # eps = 0.04
    ## --
    #
    kopt_new = 0
    fvalues = {}
    runtimes = {}
    nmis = {}
    fopts = {}
    labels_opt = np.array([])
    Y_opt = []
    list_labels = {}
    list_Y = {}
    for kk in range(kmin,kmax+1):
        t0 = timer()
        labels, _, Y = lvlearn_heteroskedastic(X, hatSigma=hatSigma, rank=kk, max_iter=200, eps=eps)
        tt = timer() - t0
        runtimes[kk] = tt
        list_labels[kk] = labels
        list_Y[kk] = Y
        C = _convert_labels2centroids(Y, labels)
        fvalues[kk] =  _SON_objective_function(C, Y, gamma=gamma)
        # Normalized Mutual Information (NMI)
        if labels_true is not None:
            nmis[kk] = normalized_mutual_info_score(labels_true, labels)
        else:
            nmis[kk] = np.nan
        #
        mini3_indices = _get_tail3( fvalues )
        kstar_ = mini3_indices[0]
        kstar = int(kstar_)
        labels_opt = list_labels[kstar]
        Y_opt = list_Y[kstar]
        fopts[kstar] = fvalues[kstar]
        if (kstar_ < kk - 3) :
            #
            if fvalues[mini3_indices[0]] < 0.1:
                kstar = mini3_indices[0]
            break
    return (labels_opt, kstar, runtimes, nmis, fopts, fvalues)

def modelSel4CLV_EV(X, hatSigma=None, gamma=0.2, eps=0.01, labels_true=None):
    n_samples, d = X.shape
    if hatSigma == None:
        Xc = X - np.mean(X, axis=0, keepdims=True)
        hatSigma = np.dot(Xc.T, Xc) / n_samples
    # Init
    (U,S,Ut) = np.linalg.svd( hatSigma )
    kmin = max(5, _topK_by_ratio(S, 0.7))
    kmax = min(d-2, max(19, _topK_by_ratio(S, 0.98)))
    #
    kstar = kmin
    fvalues = {}
    nmis = {}
    fopts = {0: np.Inf}
    labels_opt = np.array([])
    Y_opt = []
    list_labels = {}
    list_Y = {}
    for kk in range(kmin,kmax+1):
        labels, Y = lvlearn_ev2(X, SVD=(U,S), rank=kk, eps=eps )
        list_labels[kk] = labels
        list_Y[kk] = Y
        C = _convert_labels2centroids(Y, labels)
        fvalues[kk] =  _SON_objective_function(C, Y, gamma=gamma)
        # Normalized Mutual Information (NMI)
        if labels_true is not None:
            nmis[kk] = normalized_mutual_info_score(labels_true, labels)
        else:
            nmis[kk] = np.nan
        # if both median(kopt, kopt2, kopt3) is smaller than kk-3
        # then adopt the integer upper(median)
        mini3_indices = _get_tail3( fvalues )
        kstar_ = np.median( mini3_indices )
        if (kstar_ < kk - 3) :
            if not kstar_.is_integer():
                kstar = int(kstar_) + 1
            else:
                kstar = int(kstar_)
            # Exception: when min( fvalues ) is really small
            if fvalues[mini3_indices[0]] < 0.1:
                kstar = mini3_indices[0]
            labels_opt = list_labels[kstar]
            Y_opt = list_Y[kstar]
            fopts[kstar] = fvalues[kstar]
            break
    if len(labels_opt) < 1:
        print('debugging model sel function: ')
        print('kmin,kmax:', kmin, kmax)
        print(fvalues)
    return (labels_opt, kstar, nmis, fopts, fvalues)

def modelSel4CLV_NV(X, hatSigma=None, gamma=0.2, eps=0.01, labels_true=None):
    n_samples, d = X.shape
    if hatSigma is None:
        Xc = X - np.mean(X, axis=0, keepdims=True)
        hatSigma = np.dot(Xc.T, Xc) / n_samples
    # Init
    (U,S,Ut) = np.linalg.svd( hatSigma )
    kmin = max(5, _topK_by_ratio(S, 0.7))
    kmax = min(d-2, max(19, _topK_by_ratio(S, 0.98)))
    #
    kopt_new = 0
    fvalues = {}
    nmis = {}
    fopts = {0: np.Inf}
    labels_opt = np.array([])
    Y_opt = []
    list_labels = {}
    list_Y = {}
    for kk in range(kmin,kmax+1):
        labels, _, Y = lvlearn_heteroskedastic(X, hatSigma=hatSigma, rank=kk, max_iter=200, eps=eps)
        list_labels[kk] = labels
        list_Y[kk] = Y
        C = _convert_labels2centroids(Y, labels)
        fvalues[kk] =  _SON_objective_function(C, Y, gamma=gamma)
        #
        if labels_true is not None:
            nmis[kk] = normalized_mutual_info_score(labels_true, labels)
        else:
            nmis[kk] = np.nan
        mini3_indices = _get_tail3( fvalues )
        kstar_ = np.median( mini3_indices )
        if (kstar_ < kk - 3) :
            if not kstar_.is_integer():
                kstar = int(kstar_) + 1
            else:
                kstar = int(kstar_)
            #
            if fvalues[mini3_indices[0]] < 0.1:
                kstar = mini3_indices[0]
            labels_opt = list_labels[kstar]
            Y_opt = list_Y[kstar]
            fopts[kstar] = fvalues[kstar]
            break
    return (labels_opt, kstar, nmis, fopts, fvalues)
"""
    END - Model selection functions
"""

def output_scores_by_rank( output, dir_out ):
    fvals = output[-1]
    fopts = output[-2]
    nmis = output[-3]
    filename = f'{dir_out}/scores_{algo}.csv'
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        with open(filename, 'a') as f1:
            for kk in fvals:
                f1.write("%s,%d,%s,%s,%d,%d,%.3f,%d,%s,%.3f,%.3f,%.3f,%.3f\n"%(graph_type, deg, sem_type, weights_mode, \
                d,kk, rnd, SEED, '%s_%.4f'%(algo,eps), np.nan,nmis[kk], np.nan, fvals[kk]))
    else:
        with open(filename, 'w') as f1:
            f1.write('graph_type,deg,sem_type,mode_edgeWeights,d,k,rnd,seed,alg,ARI,NMI,time,SON_fvalue\n')
            for kk in fvals:
                f1.write("%s,%d,%s,%s,%d,%d,%.3f,%d,%s,%.3f,%.3f,%.3f,%.3f\n"%(graph_type, deg, sem_type, weights_mode, \
                d,kk, rnd, SEED, '%s_%.4f'%(algo,eps), np.nan,nmis[kk], np.nan,fvals[kk]))
    return 0

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
            elif p_k in ['baseline', 'algo']:
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
        'algo': 'na', \
        'LAM': 0.1, 'opts': [0.1], \
        'ipara': 0, 'fdir':'', 'fout':'', \
        'VAR': 0, 'verbo': 2, \
        }
    args = _input_args(args)
    filename = args['filename'] # output file for storing the info of MBs

    ## 1/ load or generate network and data
    # --- syn-oracle means read from synthetic data
    if args['runwho'][0] in ['syn-ev', 'syn-oracle']:
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
        opts = args['opts'] #
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
        # # options for random data

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

    #
    algo = args['algo'][0]
    print('\n\nAlgorithm to run is: ', algo)

    if len(opts) > 3:
        eps = opts[3]
    else:
        if opts[1] == 1:
            eps=3e-2
        else:
            eps=3e-3
    #
    if algo in ['lvlearn-nv']:
        tb = timer()
        labels, stats = lvlearn_heteroskedastic(X, rank=k, max_iter=200, eps=eps)
        tt=timer() - tb
        ttotal = tt
    elif algo in ['lvlearn-ev' ]:
        tb = timer()
        labels  = lvlearn_homoskedastic(X, rank=k, eps=eps)
        tt=timer() - tb
        ttotal = tt
    elif algo in ['lvlearn-EVmodelSel']:
        tb = timer()
        output = lvlearn_modelSel_EV(X, eps=eps, gamma=0.0, labels_true=labels_true)
        ttotal=timer() - tb
        labels = output[0]
        print('===== SON function values: ', output[-1])
        print('===== Optimal SON function values: ', output[-2])
        print('===== NMIs: ', output[-3])
        output_scores_by_rank(output, fdir)
        algo = '%s_%.4f'%(algo,eps)
        rank_sel = output[1].shape[1] # Y_opt column numbers
        tt = output[2][rank_sel]
    elif algo in ['lvlearn-NVmodelSel']:
        tb = timer()
        output = lvlearn_modelSel_NV(X, eps=eps, gamma=0.0, labels_true=labels_true)
        ttotal=timer() - tb
        labels = output[0]
        print('===== SON function values: ', output[-1])
        print('===== Optimal SON function values: ', output[-2])
        print('===== NMIs: ', output[-3])
        print('===== Rank chosen: ', output[1])
        output_scores_by_rank(output, fdir)
        algo = '%s_%.4f'%(algo,eps)
        #
        rank_sel = output[1] # Y_opt column numbers
        tt = output[2][rank_sel]
    elif algo in ['CLV-modelSelEV']:
        tb = timer()
        output = modelSel4CLV_EV(X, eps=eps, gamma=0.0, labels_true=labels_true)
        tt0=timer() - tb
        print('===== SON function values: ', output[-1])
        print('===== Optimal SON function values: ', output[-2])
        print('===== NMIs: ', output[-3])
        print('===== rank selected: ', output[1])
        t0 = timer()
        #--- Import CLV
        labels = CLV2( X, K=output[1] )
        #
        tt = timer() - t0
        ttotal = tt
    elif algo in ['CLV-modelSelNV']:
        tb = timer()
        output = modelSel4CLV_NV(X, eps=eps, gamma=0.0, labels_true=labels_true)
        tt0=timer() - tb
        print('===== SON function values: ', output[-1])
        print('===== Optimal SON function values: ', output[-2])
        print('===== NMIs: ', output[-3])
        print('===== rank selected: ', output[1])
        t0 = timer()
        #--- Import CLV
        labels = CLV2( X, K=output[1] )
        # labels = CLV1( X, K=output[1] )
        tt = timer() - t0
        ttotal = tt
    else:
        print('Algo not known!')
    Pi = {}
    for i in range(len(np.unique(labels))):
        Pi[i] = list(np.where(labels==i)[0])

    # Adjusted Rand Index (ARI)
    ari = adjusted_rand_score(labels_true, labels)
    print("\n(%s) Adjusted Rand Index:"%algo, ari)
    # Normalized Mutual Information (NMI)
    nmi = normalized_mutual_info_score(labels_true, labels)
    print("(%s) Normalized Mutual Information:"%algo, nmi)

    # --- Evaluate results
    noise_var = opts[2]
    runwho = args['runwho'][0]
    print('\n\ngraph_type,deg,sem_type,mode_edgeWeights,evnv,noise_var,d,k,rnd,seed,alg,ARI,NMI,time,time_total')
    print("%s,%d,%s,%s,%s,%.2f,%d,%d,%.3f,%d,%s,%.3f,%.3f,%.3f,%.3f"%(graph_type, deg, sem_type, weights_mode, runwho, noise_var, \
            d,k, rnd, SEED, algo, ari,nmi, tt, ttotal))


