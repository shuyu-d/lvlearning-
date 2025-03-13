import sys
sys.path.append("..")
import numpy as np
import pandas as pd
import itertools
from aux import utils
import random

def gen_list_settings(graph_types=None, sem_types=None, degs=[1.0], d=[100], n=[200]):
    if graph_types is None:
        graph_types = ['ER']
    if sem_types is None:
        sem_types = ['gauss']

    l_p = list(itertools.product(d, graph_types, \
                             sem_types, \
                             degs, n))
    df_p = pd.DataFrame(l_p, columns=['d','graph_type',\
                                        'sem_type', \
                                        'deg', 'n'])
    return l_p, df_p

def gen_list_optparams(opts=None):
    if opts is None:
        opts={'k': [25], 'lambda_1': [2e-1,4e-1], 'idec_lambda1':[2e-1]}
    nkeys = len(opts.items())
    ll = []
    cols = []
    for key, li in opts.items():
        ll.append(li)
        cols.append(key)
    l_p = list(itertools.product(*ll))
    df_p = pd.DataFrame(l_p, columns=cols)
    return l_p, df_p

def gen_data_sem(d=100,deg=1.0,n=200,graph_type='ER',sem_type='gauss', seed=1):
    utils.set_random_seed(seed)
    s0 = int(deg*d)
    B_true = utils.simulate_dag(d, s0, graph_type)
    if graph_type is 'SF':
        # In the case of scale-free (SF) graphs, hubs are mostly
        # causes, rather than effects, of its neighbors
        B_true = B_true.T
    W_true = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(W_true, n, sem_type)
    return W_true, X

def gen_data_sem_original(d=100,deg=1.0,n=200,graph_type='ER',sem_type='gauss', seed=1, w_ranges=((-2.0, -0.5), (0.5, 2.0)) ):
    utils.set_random_seed(seed)
    s0 = int(deg*d)
    B_true = utils.simulate_dag(d, s0, graph_type)
    # NOTE [#923-2024]: enabled the optional input args 'w_ranges' (this argument is the original one in utils.py)
    W_true = utils.simulate_parameter(B_true, w_ranges=w_ranges)
    X = utils.simulate_linear_sem(W_true, n, sem_type)
    return W_true, X


def gen_data_sem_nv(d=100,deg=1.0,n=200,graph_type='ER',sem_type='gauss', \
        seed=1, w_ranges=((-2.0, -0.5), (0.5, 2.0)), noise_scale=None):
    utils.set_random_seed(seed)
    s0 = int(deg*d)
    B_true = utils.simulate_dag(d, s0, graph_type)
    #
    W_true = utils.simulate_parameter(B_true, w_ranges=w_ranges)
    X = utils.simulate_linear_sem(W_true, n, sem_type, noise_scale=noise_scale)
    return W_true, X



def gen_data_sem_notriangle(d=100,deg=1.0,n=200,graph_type='ER',sem_type='gauss', seed=1):
    utils.set_random_seed(seed)
    s0 = int(deg*d)
    B_pre = utils.simulate_dag(d, s0, graph_type)
    # ---NOTE: add a function that prunes any edge in a triangle without
    #          changing the moral graph of the pruned DAG
    B_true = prune_triangles(B_pre)
    #----
    W_true = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(W_true, n, sem_type)
    return W_true, X


def gen_graph_dag(d=100,deg=1.0,graph_type='ER', seed=1):
    utils.set_random_seed(seed)
    s0 = int(deg*d)
    B_true = utils.simulate_dag(d, s0, graph_type)
    if graph_type is 'SF':
        # In the case of scale-free (SF) graphs, hubs are mostly
        # causes, rather than effects, of its neighbors
        B_true = B_true.T
    W_true = utils.simulate_parameter(B_true)
    return W_true

def get_markovblanket(B, node_index=None):
    if node_index is None:
        node_index = range(B.shape[0])
    mb = []
    for i in node_index:
        # descendants
        j_desc = np.array((B[i,:] != 0))
        # ascendants
        j_asc = np.array((B[:,i] != 0))
        # spouses
        j_epou = np.zeros(j_desc.shape)
        node_desc = np.where((B[i,:] != 0))[0]
        if len(node_desc) > 0:
            j_epou = np.array((B[:,node_desc[0]]!=0))
            for ii in node_desc:
                j_epou += np.array((B[:,ii]!=0))
        # # MB of node i
        # print('j-epoux set is:')
        # print(j_epou) # (d,0)
        # print('j-desc are:')
        # print(j_desc) # (d,)
        mb.append((j_desc + j_asc + j_epou)>0) # append one 1d array
    return mb

def gen_graph_dag_with_markovblanket(d=100,deg=1.0,graph_type='ER', seed=1):
    utils.set_random_seed(seed)
    s0 = int(np.ceil(deg*d))
    B_true = utils.simulate_dag(d, s0, graph_type)
    if graph_type is 'SF':
        # In the case of scale-free (SF) graphs, hubs are mostly
        # causes, rather than effects, of its neighbors
        B_true = B_true.T
    # Get the Markov blanket of a given node (or every node)
    mb = get_markovblanket(B_true, node_index=None)
    W_true = utils.simulate_parameter(B_true)
    return W_true, mb

def gen_graph_dag_tril(d=100,deg=1.0,graph_type='ER', seed=1, format_tril=True):
    utils.set_random_seed(seed)
    s0 = int(np.ceil(deg*d))
    if format_tril:
        B_true = utils.simulate_dag_tril(d, s0, graph_type)
    else:
        B_true = utils.simulate_dag(d, s0, graph_type)
    if graph_type != 'ER':
        print('DAG in triangular form only supported for ER graphs')
    W_true = utils.simulate_parameter(B_true)
    return W_true

def prune_triangles(B):
    # 1. local all v-structures
    # 2. whenever there is a link between the two parent nodes in the
    #      v-structure, prune that link
    print('Orignal DAG generated: see readme.txt.\n Start pruning edges that exist between any pair of parents in a v-structure: \n')
    torm = set()
    for k in range(B.shape[0]):
        pp = np.where(B[:,k]!=0)[0]
        pairs_nested = [(i, j) for i in pp for j in pp if i < j]
        # (i,j;k) forms a v-structure
        torm = set.union(torm, set( pairs_nested )  )
    print(torm)
    for (i,j) in list(torm):
        B[i,j] = 0
        B[j,i] = 0
    print('\nNow the resulting DAG should no longer have triangles in its skeleton.\n\nNOTE: the resulting DAG adjacency matrix is stored in .txt and .npz formats, under "notriangle_seed*/"\n\n')
    return B

def gen_bipartite_latent2observed(n_latent_pre = 10, n_observed = 50,
                                  cluster_max = 10 , verbo=0):
    """
        1-Factor LV model satisfying Assumption 2.
        For each latent variable j in [k], draw n_j numbers from [d] without
        replacement.
    """
    I = set( range(n_observed) )
    Pi = {}
    while len(I) > 0:
        for j in range(n_latent_pre):
            if len(I) >= 3:
                n_j = random.randint(3, cluster_max)
                if len(I) >= n_j :
                    inds = random.sample( I, n_j )
                    if j not in Pi:
                        Pi[j] = inds
                    else:
                        Pi[j] += inds
                    I = I - set(inds)
                    if verbo > 1:
                        print('Just created Pi[%d] now. '% j)
                        print('I is: ', I)
                        print('Pi is: ', Pi)
                else:
                    if j not in Pi:
                        Pi[j] = list(I)
                    else:
                        Pi[j] += list(I)
                    I = I - I
                    if verbo > 1:
                        print('Just created Pi[%d] with the rest. '% j)
                        print('I is: ', I)
                        print('Pi is: ', Pi)
            else:
                # randomly select a latent variable
                J_active = list( Pi.keys() )
                jstar = random.sample(J_active, 1)[0]
                Pi[jstar] += list(I)
                I = I - I
                if verbo > 1:
                    print('Append the rest to Pi[%d]. '% jstar)
                    print('I is: ', I)
                    print('Pi is: ', Pi)
            # Stopping criterion
            if len(I) == 0:
               break
    return Pi
def convert_clustering2adjmatrix(Pi, n_observed=None, mode_weights=1):
    # (Optional) Verification of Pi and n_observed.
    if n_observed == None:
        n_observed = sum( [len(Pi[j]) for j in Pi] )
    else:
        if n_observed != sum( [len(Pi[j]) for j in Pi] ):
            print('Pi is a valid clustering regarding the input n_observed.')
        else:
            print('Pi is NOT valid regarding the input n_observed!')
    index_max = max( [ max(Pi[j]) for j in Pi ] )
    if index_max == n_observed - 1:
        print('Pi is a valid clustering.')
    else:
        print('Pi is not consistent!')
    # Convert Pi to (0,1)-valued adjacency matrix Gamma
    k = len( Pi.keys() )
    Gamma = np.zeros( [n_observed, k] )
    for j in range(k):
        Gamma[Pi[j], j] = 1
    # NOTE: the above setting makes B totally compliant with the conditions for
    # perfect recovery of the clusters via the SON convex clustering.
    W = Gamma.copy()

    if mode_weights == 1: #False:
        # wB1
        wr = ((-2, -0.5), (0.5, 2))
        W = np.zeros(Gamma.shape)
        S = np.random.randint(len(wr), size=Gamma.shape)  # which range
        for i, (low, high) in enumerate(wr):
            uni = np.random.uniform(low=low, high=high, size=Gamma.shape)
            W += Gamma * (S == i) * uni
    elif mode_weights == 2:
        # wB2
        wr = (0.5, 1.5)
        W = np.zeros(Gamma.shape)
        uni = np.random.uniform(low=wr[0], high=wr[1], size=Gamma.shape)
        W += Gamma * uni
    return W

def simulate_data_linearLVM(d, n_samples, k, opts, SEED, \
                            deg=2, graph_type='ER', sem_type='gauss'):
    # d : number of observed variables
    # k : number of latent variables

    # Generate random graph and observational data
    # ---
    if opts[1] == 1:
        weights_mode = '+/-'
    elif opts[1] == 2:
        weights_mode = '+only'
    else:
        weights_mode = 'Binary'
    print('Generating a DAG among %d latent variables: ' %k)
    utils.set_random_seed(SEED)
    s0 = int(deg*k)
    # C_true = utils.simulate_dag(k, s0, graph_type)
    C_true = utils.simulate_dag_tril(k, s0, graph_type)
    C_true = C_true.T
    Cw_true = utils.simulate_parameter(C_true)
    # --- Generate a bipartite graph from [k] to [d]
    # Idea: for each j in [k], randomly select more than 3 nodes from [d]
    # without replacement until all nodes are attributed. Remove the last
    # one latent variable that does not have at least 3 measured variables.
    Pi_true = gen_bipartite_latent2observed(n_latent_pre = k, n_observed = d, cluster_max = opts[0])
    Bw_true = convert_clustering2adjmatrix(Pi_true, mode_weights=opts[1])
    B_true = (Bw_true != 0)  # (0,1)-valued matrix of size d x k

    noise_var = opts[2]
    Y_latent = utils.simulate_linear_sem(Cw_true, n_samples, sem_type)
    X_pre = utils.simulate_linear_sem(np.zeros([d, d]), n_samples, sem_type, noise_scale=noise_var)
    #
    X = Y_latent @ (Bw_true.T) + X_pre
    # Convert pi_true to labels
    labels_true = np.array([0 for _ in range(d)])
    for j in Pi_true:
        for ii in Pi_true[j]:
            labels_true[ii] = j
    return X, Cw_true, Bw_true, Pi_true, labels_true, Y_latent, X_pre, weights_mode

def simulate_data_linearLVM_NV(d, n_samples, k, opts, SEED, \
                              deg=2, graph_type='ER', sem_type='gauss'):
    # d : number of observed variables
    # k : number of latent variables

    # Generate random graph and observational data
    # ---
    if opts[1] == 1:
        weights_mode = '+/-'
    elif opts[1] == 2:
        weights_mode = '+only'
    else:
        weights_mode = 'Binary'
    print('Generating a DAG among %d latent variables: ' %k)
    utils.set_random_seed(SEED)
    s0 = int(deg*k)
    # C_true = utils.simulate_dag(k, s0, graph_type)
    C_true = utils.simulate_dag_tril(k, s0, graph_type)
    C_true = C_true.T
    Cw_true = utils.simulate_parameter(C_true)
    # --- Generate a bipartite graph from [k] to [d]
    # Idea: for each j in [k], randomly select more than 3 nodes from [d]
    # without replacement until all nodes are attributed. Remove the last
    # one latent variable that does not have at least 3 measured variables.
    Pi_true = gen_bipartite_latent2observed(n_latent_pre = k, n_observed = d, cluster_max = opts[0])
    Bw_true = convert_clustering2adjmatrix(Pi_true, mode_weights=opts[1])
    B_true = (Bw_true != 0)  # (0,1)-valued matrix of size d x k

    noise_var = opts[2]
    Y_latent = utils.simulate_linear_sem(Cw_true, n_samples, sem_type)
    nv_vec = np.random.uniform(low=1.0, high=noise_var, size=[d,])
    nv_vec[ nv_vec > 0.9*noise_var ] = 6
    X_pre = utils.simulate_linear_sem(np.zeros([d, d]), n_samples, sem_type, noise_scale=nv_vec)
    #
    X = Y_latent @ (Bw_true.T) + X_pre
    # Convert pi_true to labels
    labels_true = np.array([0 for _ in range(d)])
    for j in Pi_true:
        for ii in Pi_true[j]:
            labels_true[ii] = j
    return X, Cw_true, Bw_true, Pi_true, labels_true, Y_latent, X_pre, weights_mode

if __name__ == '__main__':

    # l_p, df_p = gen_list_settings(n=[200,400,600,800])

    # for v in l_p:
    #     print(v)
    # print(df_p)

    Wt, mb = gen_graph_dag_with_markovblanket(d=100,deg=1.0,graph_type='ER', seed=1)
    print(Wt)
    print(np.where(mb[0]))

    d, deg, gt, seed = 100, 5.0, 'ER', 3
    Wt = gen_graph_dag_tril(d=d, deg=deg, graph_type=gt, seed=seed)

