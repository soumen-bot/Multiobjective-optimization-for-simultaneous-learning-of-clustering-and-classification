'''Kernelized General Fuzzy c-Means Clustering

Paper Source: Gupta A., Das S., 'On the Unification of k-Harmonic Means and
Fuzzy c-Means Clustering Problems under Kernelization', in the 2017 Ninth
International Conference on Advances in Pattern Recognition (ICAPR 2017),
pp. 386-391, 2017.
'''

# Author: Avisek Gupta


import numpy as np
import copy
from collections import Counter
from scipy.spatial.distance import cdist


def  RKFCM_MSCC(
    X, Y, centers, n_clusters, n_classes, _lambda, m=2, p=2, max_iter=300,
    n_init=30, tol=1e-16
):
    '''Kernelized General Fuzzy c-Means Clustering

    Notes
    -----
    Set p=2 for Kernel Fuzzy c-Means.
    Set m=2 for Kernel k-Harmonic Means.

    Parameters
    ----------

    X : array, shape (n_data_points, n_features)
        The data array.

    Y : array, shape (n_data_points, 1)
        The actual classes array.

    centers : array, shape (n_clusters, n_features)
        particle positions for MOPSO.

    n_clusters : int
        The number of clusters.

    n_classes : int
        The number of classes.

    _lambda : float
        Used in calculation of sigma

    m : float, default: 2
        Level of fuzziness. Set m=2 for Kernel k-Harmonic Means.

    p: float, default: 2
        Power of Euclidean distance. Set p=2 for Kernel Fuzzy c-Means.

    max_iter: int, default: 300
        The maximum number of iterations of the KGFCM algorithm.

    n_init: int, default 30
        The number of runs of the KGFCM algorithm. The results corresponding
        to the minimum cost will be returned.

    tol: float, default 1e-16
        Tolerance for convergence.

    Returns
    -------

    mincost_centers: array, shape (n_clusters, n_features)
        The resulting cluster centers.

    min_cost: float
        The lowest cost achieved in n_init runs.

    misclassification_rate:
        The resulting value of J2(misclassification rate)
    
    P:
        The resulting P relation matrix

    '''

    N = X.shape[0]
    min_cost = +np.inf

    '''
        sigma2 : sigma ** 2
            Parameter for the Gaussian Kernel.
            K(a,b) = np.exp(-(a-b)**2 / sigma2)

    '''
    sigma2 = (np.power((X-(X.mean(0))), 2).sum(1).max())/_lambda
    
    for _ in range(n_init):
    
        for v_iter in range(max_iter):

            K = np.exp(
                -cdist(centers, X, metric='sqeuclidean') / ((sigma2))
            )
            K_dist = np.fmax(1 - K, np.finfo(np.float64).eps)

            # Update memberships
            U = np.fmax(
                K_dist ** (-p / (2 * (m - 1))), np.finfo(np.float64).eps
            )
            U = U / U.sum(axis=0)

            # Update centers
            old_centers = np.array(centers)
            expr_part = np.fmax((
                (U ** m) * (K_dist ** ((p - 2) / 2)) * K
            ), np.finfo(np.float64).eps)
            centers = expr_part.dot(X) / expr_part.sum(axis=1)[:, None]

            if ((centers - old_centers) ** 2).sum() < tol:
                break

        # Compute cost
        cost = 2*(((U ** m) * (K_dist ** (p / 2))).sum())

        if cost < min_cost:

            min_cost = cost
            mincost_centers = np.array(centers)
            mincost_mem = U.argmax(axis=0)

    P = np.zeros((n_clusters, n_classes))   # P relation matrix

    Wl_Ck_freq = Counter(map(tuple, zip(mincost_mem,Y)))    # Num(x∈𝑤𝑙 and x∈𝑐𝑘)

    for key in Wl_Ck_freq:
        P[key] = copy.deepcopy(Wl_Ck_freq[key])

    Ck_freq = copy.deepcopy(np.bincount(mincost_mem,minlength=n_clusters))  # Num(x∈𝑐𝑘)

    Ck_freq[Ck_freq==0]=1
    P =copy.deepcopy( P / Ck_freq[:,None])  # P(𝑤𝑙|𝑐𝑘) = Num(x∈𝑤𝑙 and x∈𝑐𝑘)/Num(x∈𝑐𝑘)

    P_Wl_Xi = np.matmul(np.transpose(U),P)  # P(𝑤𝑙|xi)
    
    f_Xi = P_Wl_Xi.argmax(axis=1)

    misclassification_rate = np.sum(f_Xi!=Y) / N
    
    return mincost_centers, min_cost, misclassification_rate, P 


