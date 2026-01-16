import os 
import numpy as np
from scipy import stats
from neuromaps import nulls

def get_spin_nulls(spin_path, atlas_tuple=None, n_perm=10000, seed=1024, nroi=210):
    if not os.path.exists(spin_path):
        if atlas_tuple is None:
            raise  ValueError(f"The file {spin_path} does not exist. An atlas file must be provided to generate spin nulls. \nThe current parameter atlas_tuple is None.")
        else:
            print('Generating spin nulls ...')
            rotated = nulls.vasa(np.arange(nroi), atlas='fsLR', density='32k', n_perm=n_perm, seed=seed, parcellation=atlas_tuple, spins=None)
            np.save(spin_path, rotated.astype(int))
    

def get_CoDE_stat(sc, brainmap, spin_path, atlas_gii=None, norm_type=None, n_perm=10000, seed=1024):
    
    assert sc.shape[0] == brainmap.size
    
    if norm_type is None:
        brainmap = brainmap.reshape(1, -1)
    elif norm_type == 'zscore':
        brainmap = (brainmap - np.mean(brainmap)) / np.std(brainmap)
        brainmap = brainmap.reshape(1, -1)
    else:
        raise NotImplementedError
    
    get_spin_nulls(spin_path, atlas_tuple=atlas_gii, n_perm=n_perm, seed=seed, nroi=sc.shape[0])

    emp_embedd = cal_CoDE(sc, brainmap)[0]
    rotated_map = nulls.vasa(brainmap[0, :], atlas='fsLR', density='32k', n_perm=n_perm, spins=spin_path, seed=None, parcellation=atlas_gii)
    perm_embedd = cal_CoDE(sc, rotated_map.T)
    p = (np.sum(emp_embedd < perm_embedd)) / n_perm
    CoDE = (emp_embedd - np.mean(perm_embedd)) / np.std(perm_embedd)
    return CoDE, p


def normalized_laplacian(adj_matrix):
    R = np.sum(adj_matrix, axis=1) + 1e-12
    R_sqrt = 1/np.sqrt(R)
    D_sqrt = np.diag(R_sqrt)
    I = np.eye(adj_matrix.shape[0])
    return I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)
    

def cal_CoDE(A, X, mode='sysN', precomputed=False):
    '''
    x: N_item * N_roi
    '''
    if not precomputed:
        L = normalized_laplacian(A)
    else:
        L = A
    embedd = 1 - np.diag(np.matmul(np.matmul(X, L), X.T)) / np.diag(np.matmul(X, X.T))
    return embedd


def perm_correlation(x, y, perm_num, type='pearson'):
    perm_map = np.zeros((x.shape[0], perm_num))
    for i in range(perm_num):
        perm_map[:, i] = np.random.permutation(y)
    # perm_map = y.reshape(-1)[spins]

    if type == 'pearson':
        corr_func = stats.pearsonr
    elif type == 'spearman':
        corr_func = stats.spearmanr
    
    r = corr_func(x.reshape(-1), y.reshape(-1))[0]

    r_perm = np.zeros((perm_num))
    for i in range(perm_num):
        r_perm[i] = corr_func(x, perm_map[:, i])[0]
    
    p = (np.count_nonzero(abs(r_perm - np.mean(r_perm)) > abs(r - np.mean(r_perm)))) / perm_num
    
    return r, p
  
