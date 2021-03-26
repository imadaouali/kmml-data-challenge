import numpy as np
from tqdm import tqdm
from scipy import sparse
from itertools import combinations, product

def generate_mismatch(s, mismatch=2):
    N = len(s)
    letters = 'ACGT'
    pool = list(s)

    for indices in combinations(range(N), mismatch):
        for replacements in product(letters, repeat=mismatch):
            skip = False
            for i, a in zip(indices, replacements):
                if pool[i] == a: skip = True
            if skip: continue

            keys = dict(zip(indices, replacements))
            yield ''.join([pool[i] if i not in indices else keys[i] 
                           for i in range(N)])


def mismatch_kernel(X, length, mismatch):
    '''
    Returns the Mismatch Kernel associated to a data matrix X.
    Parameters
    ----------
    X : nd-array
        a 2d array containing the data in rows
    length : int
        the length of the subsequences to consider
    mismatch : int
        the number of mismatches to consider. If mismatch = 0 : Spectrum Kernel]
    Returns
    -------
    The Gram matrix (X.shape[0], X.shape[0])
    '''
    all_sequences_index = {}
    id_last_seq = 0
    dict_seq = {}
    print('========= Computing possible subsequences with mismatches ============ ')
    for idx in range(len(X)):
        data = X[idx]
        for i in range(len(data)-length + 1):
            seq = data[i:i+length]
            if seq not in all_sequences_index:
                all_sequences_index[seq] = id_last_seq
                id_last_seq += 1

            if mismatch >= 1:
                list_mismatch = list(generate_mismatch(seq, mismatch))
                dict_seq[seq] = list_mismatch
                for seq_mis in dict_seq[seq]:
                    if seq_mis not in all_sequences_index:
                        all_sequences_index[seq_mis] = id_last_seq
                        id_last_seq += 1

    features = sparse.lil_matrix((len(X), len(all_sequences_index)), dtype=float)

    for idx in tqdm(range(len(X))):
        data = X[idx]
        for i in range(len(data)-length + 1):
            seq = data[i:i+length]
            features[idx, all_sequences_index[seq]] += 1
            
            if mismatch >= 1:
                for seq_mis in dict_seq[seq]:
                    features[idx, all_sequences_index[seq_mis]] += (1/2)**mismatch ## Adding a penalty of (1/2)**mismatch
            
    ## Converting to sparse format before computing
    features.tocsr()
    return np.array(np.dot(features, features.T).todense())


def string_kernel(s, t, k=3, delta=0):
    """ Basic string kernel with displacement assuming equal lengths. """
    L = len(s)
    return sum(((s[i:i + k] == t[d + i:d + i + k])
                for i, d in it.product(range(L - k + 1), range(-delta, delta + 1))
                if i + d + k <= L and i + d >= 0))


def string_kernel_Gram(S, **kwargs):
    """ Compute the kernel matrix between all pairs of strings in a list S"""
    N = len(S)
    K = np.zeros((N, N))
    for i in range(N):
        K[i, i] = string_kernel(S[i], S[i], **kwargs)
    for i, j in it.combinations(range(N), 2):
        K[i, j] = K[j, i] = string_kernel(S[i], S[j], **kwargs)
    return K

def normalize_gram(K):
    DIAG = np.diag(K)
    DIAG = DIAG.reshape((-1, 1)).dot(DIAG.reshape((1, -1)))
    return K/np.sqrt(DIAG)