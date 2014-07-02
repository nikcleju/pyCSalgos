"""
tst.py

Provides Two Stage Thresholding (TST)
"""

# Author: Nicolae Cleju
# License: BSD 3 clause

import math

import numpy as np

from base import SparseSolver

class TwoStageThresholding(SparseSolver):
    """
    """

    def __init__(self, stoptol, maxiter=300, algorithm="recommended"):

        # parameter check
        if stoptol < 0:
            raise ValueError("stopping tolerance is negative")
        if maxiter <= 0:
            raise ValueError("number of iterations is not positive")

        self.stoptol = stoptol
        self.maxiter = maxiter
        self.algorithm = algorithm

    def __str__(self):
        return "TST ("+str(self.stoptol)+" | " + str(self.maxiter) + ", " + str(self.algorithm)+")"

    def solve(self, data, dictionary):
        return two_stage_thresholding(data, dictionary, self.stoptol, self.maxiter, self.algorithm)

def two_stage_thresholding(data, dictionary, stoptol, maxiter, algorithm="recommended"):

    # Force data 2D
    if len(data.shape) == 1:
        data = np.atleast_2d(data)
    if data.shape[0] < data.shape[1]:
        data = np.transpose(data)

    N = dictionary.shape[1]
    Ndata = data.shape[1]
    coef = np.zeros((N, Ndata))

    if algorithm == "recommended":
        for i in range(Ndata):
            coef[:,i] = _tst_recommended(dictionary, data[:,i], maxiter, stoptol)
    else:
        raise ValueError("Algorithm '%s' does not exist", algorithm)

    return np.squeeze(coef)




def _tst_recommended(X, Y, nsweep=300, tol=0.00001, xinitial=None, ro=None):

    colnorm = np.mean(np.sqrt((X**2).sum(0)))
    X = X / colnorm
    Y = Y / colnorm
    [n,p] = X.shape
    delta = float(n) / p

    if xinitial is None:
        xinitial = np.zeros(p)
    if ro == None:
        ro = 0.044417*delta**2 + 0.34142*delta + 0.14844

    k1 = int(math.floor(ro*n))
    k2 = int(math.floor(ro*n))

    #initialization
    x1 = xinitial.copy()
    I = []

    for sweep in np.arange(nsweep):
        r = Y - np.dot(X,x1)
        c = np.dot(X.T, r)
        i_csort = np.argsort(np.abs(c))
        I = np.union1d(I , i_csort[-k2:])

        # Make sure X[:,np.int_(I)] is a 2-dimensional matrix even if I has a single value (and therefore yields a column)
        if I.size is 1:
            a = np.reshape(X[:,np.int_(I)],(X.shape[0],1))
        else:
            a = X[:,np.int_(I)]
        xt = np.linalg.lstsq(a, Y)[0]
        i_xtsort = np.argsort(np.abs(xt))

        J = I[i_xtsort[-k1:]]
        x1 = np.zeros(p)
        x1[np.int_(J)] = xt[i_xtsort[-k1:]]
        I = J.copy()
        if np.linalg.norm(Y-np.dot(X,x1)) / np.linalg.norm(Y) < tol:
            break

    return x1.copy()


