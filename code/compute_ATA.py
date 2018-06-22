import numpy as np

# wrappers for BLAS routines
import scipy.linalg.blas as bla

def compute_v(N):
    '''
    Compute a random vector v with N elements.
    '''
    v = np.cos(np.linspace(0, 2*np.pi, N))
    return v


def compute_A(N):
    '''
    Compute a random matrix A with N x N elements.
    '''
    A = []
    for i in range(N):
        A.append(compute_v(N))
    return np.array(A)


def compute_ATA_numpy_outer(N):
    '''
    Compute ATA by using the numpy.outer.
    '''
    ATA = np.zeros((N,N))
    for i in range(N):
        ai = compute_v(N)
        # I do not understand the difference between
        # np.multiply.outer and np.outer
        # ATA += np.multiply.outer(ai, ai)
        ATA += np.outer(ai, ai)
    return ATA


def compute_ATA_blas_dger(N):
    '''
    Compute ATA by using the scipy wrraper for BLAS dger.
    '''
    ATA = np.zeros((N,N))
    for i in range(N):
        ai = compute_v(N)
        ATA = bla.dger(alpha=1, x=ai, y=ai, a=ATA, overwrite_a=1)
    return ATA


def compute_ATA_blas_dsyr(N):
    '''
    Compute only the lower or upper part of ATA
    by using the scipy wrraper for BLAS dsyr.
    '''
    ATA = np.zeros((N,N))
    for i in range(N):
        ai = compute_v(N)
        ATA = bla.dsyr(alpha=1, x=ai, lower=1, n=N, a=ATA, overwrite_a=1)
    return ATA
