"""
amp.py

Provides Approximate Message Passing (AMP), following the Matlab implementation of Ulugbek Kamilov.
"""

# Author: Nicolae Cleju
# License: BSD 3 clause

import math

import numpy as np

from base import SparseSolver

class ApproximateMessagePassing(SparseSolver):
    """
    Approximate Message Passing
    """

    def __init__(self, stoptol, maxiter=300, debias=True):

        # parameter check
        if stoptol < 0:
            raise ValueError("stopping tolerance is negative")
        if maxiter <= 0:
            raise ValueError("number of iterations is not positive")

        self.stoptol = stoptol
        self.maxiter = maxiter
        self.debias = debias

    def __str__(self):
        return "AMP ("+str(self.stoptol)+" | " + str(self.maxiter) + ")"

    def solve(self, data, dictionary, realdict=None):

        # DEBUG: normalize to avoid convergence problems
        norm = np.linalg.norm(dictionary)
        dictionary = dictionary/norm
        data = data/norm

        # Force data 2D
        if len(data.shape) == 1:
            data = np.atleast_2d(data)
            if data.shape[0] < data.shape[1]:
                data = np.transpose(data)

        N = dictionary.shape[1]
        Ndata = data.shape[1]
        coef = np.zeros((N, Ndata))

        for i in range(Ndata):
            coef[:,i] = _amp(dictionary, data[:,i], tol=self.stoptol, maxiter=self.maxiter)

            # Debias
            if self.debias == True:
                # keep first measurements/2 atoms
                cnt = int(round(data.shape[0]/2.0)) # how many atoms to keep
                srt = np.sort(np.abs(coef[:,i]))[::-1]
                thr = (srt[cnt-1] + srt[cnt])/2.0  # required threshold
                supp = (np.abs(coef[:,i]) > thr)
            elif self.debias == "real":
                cnt = realdict['support'].shape[0]
                srt = np.sort(np.abs(coef[:,i]))[::-1]
                thr = (srt[cnt-1] + srt[cnt])/2.0  # required threshold
                supp = (np.abs(coef[:,i]) > thr)
            elif self.debias == "all":
                cnt = data.shape[0]
                srt = np.sort(np.abs(coef[:,i]))[::-1]
                thr = (srt[cnt-1] + srt[cnt])/2.0  # required threshold
                supp = (np.abs(coef[:,i]) > thr)
            elif isinstance(self.debias, (int, long)):
                # keep specified number of atoms
                srt = np.sort(np.abs(coef[:,i]))[::-1]
                thr = (srt[self.debias-1] + srt[self.debias])/2.0  # required threshold
                supp = (np.abs(coef[:,i]) > thr)
            elif isinstance(self.debias, float):
                # keep atoms larger than threshold
                supp = (np.abs(coef[:,i]) > self.debias)
            elif self.debias != False:
                raise ValueError("Wrong value for debias paramater")

            if self.debias is not False and np.any(supp):
                gamma2 = np.zeros_like(coef[:,i])
                gamma2[supp] = np.dot( np.linalg.pinv(dictionary[:, supp]) , data[:, i])
                gamma2[~supp] = 0
                # Rule of thumb check is debiasing went ok: if very different
                #  from original gamma, debiasing is likely to have gone bad
                #if np.linalg.norm(coef[:,i] - gamma2) < 2 * np.linalg.norm(coef[:,i]):
                #    coef[:,i] = gamma2
                coef[:,i] = gamma2
                    #else:
                    # leave coef[:,i] unchanged

        return coef

def _amp(dictionary, measurements, tol=0.00001, maxiter=500):

    [n,N] = dictionary.shape

    # Initial solution
    xhat = np.zeros(N)
    z = measurements

    # Start estimation
    for t in range(maxiter):
        # Pre-threshold value
        gamma = xhat + np.dot(dictionary.T, z)

        # Find n-th largest coefficient of gamma
        threshold = largestElement(np.abs(gamma), n)

        # Estimate the signal (by soft thresholding)
        xhat = eta(gamma, threshold)

        # Update the residual
        z = measurements - np.dot(dictionary, xhat) + (z/n)*np.sum(etaprime(gamma, threshold))

        # Stopping criteria
        if(np.linalg.norm(measurements - np.dot(dictionary,xhat), 2)/np.linalg.norm(measurements,2) < tol):
            break

    return xhat


def largestElement(x, n):
    """
    Returns the n-th largest element of x
    :param x:
    :param n:
    :return:
    """

    # The n-th largest element is the (N+1-n)th smallest element
    len = x.size
    desc_pos = (len-n)
    return np.partition(x, desc_pos)[desc_pos]

def eta(x, threshold):
    # ETA performs a soft thresholding on the input x.
    return softthresh(x, threshold)

def etaprime(x, threshold):
    # ETAPRIME is the derivative of the soft threshold function. In reality it
    # returns 0 if x in [-threshold, threshold] and 1 otherwise.
    return (x > threshold) + (x < -threshold)

def softthresh(data, value, substitute=0):
    """
    Soft thresholding
    :param data:
    :param value:
    :param substitute:
    :return:
    """
    # assume value > 0
    outdata = data.copy()
    outdata[(data<=value) & (data>=-value)] = substitute
    outdata[data>value] -= value
    outdata[data<-value] += value
    return outdata