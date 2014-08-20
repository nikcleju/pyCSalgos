"""
sl0.py

Provides the Smoothed-L0 (SL0) algorithm
"""

import numpy as np

from base import SparseSolver


class SmoothedL0(SparseSolver):

    # All parameters related to the algorithm itself are given here.
    # The data and dictionary are given to the run() method
    def __init__(self, stopval, algorithm="exact"):

        # parameter check
        if stopval <= 0:
            raise ValueError("sigmamin is negative or zero")

        self.stopval = stopval
        self.algorithm = algorithm

    def __str__(self):
        return "SmoothedL0 ("+str(self.stopval)+", "+str(self.algorithm)+")"


    def solve(self, data, dictionary, realdict=None):
        return sl0(data, dictionary, self.stopval, self.algorithm)


def sl0(data, dictionary, sigmamin=1e-6, algorithm="exact"):

    if sigmamin <= 0:
        raise ValueError("sigmamin is negative or zero")

    # Force data 2D
    if len(data.shape) == 1:
        data = np.atleast_2d(data)
        if data.shape[0] < data.shape[1]:
            data = np.transpose(data)

    N = dictionary.shape[1]
    Ndata = data.shape[1]
    coef = np.zeros((N, Ndata))

    if algorithm == "exact":
        for i in range(Ndata):
                coef[:, i] = sl0_exact(dictionary, data[:,i], sigmamin)
    else:
        raise ValueError("Algorithm '%s' does not exist", algorithm)
    return np.squeeze(coef)


def sl0_exact(A, x, sigma_min, sigma_decrease_factor=0.5, mu_0=2, L=3, A_pinv=None, true_s=None):
    """
    Original matlab code and web-page:
        http://ee.sharif.ir/~SLzero
    """

    if A_pinv is None:
        A_pinv = np.linalg.pinv(A)

    if true_s is not None:
        ShowProgress = True
    else:
        ShowProgress = False

    # Initialization
    s = np.dot(A_pinv,x)
    sigma = 2.0*np.abs(s).max()

    # Main Loop
    while sigma>sigma_min:
        for i in np.arange(L):
            delta = s * np.exp( (-np.abs(s)**2) / sigma**2) # old function OurDelta()
            s = s - mu_0*delta
            s = s - np.dot(A_pinv,(np.dot(A,s)-x))   # Projection

        if ShowProgress:
            string = '     sigma=%f, SNR=%f\n' % sigma, _estimate_SNR(s,true_s)
            print string

        sigma = sigma * sigma_decrease_factor

    return s

def _estimate_SNR(estim_s, true_s):

    err = true_s - estim_s
    return 10*np.log10((true_s**2).sum()/(err**2).sum())

