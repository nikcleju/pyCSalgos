"""
uap.py

Provides Unconstrained Analysis Pursuit(GAP) for analysis-based recovery
"""

# Author: Nicolae Cleju
# License: BSD 3 clause

import math
import scipy
import numpy as np

from base import AnalysisSparseSolver
from utils import fast_lstsq


class UnconstrainedAnalysisPursuit(AnalysisSparseSolver):

    # All parameters related to the algorithm itself are given here.
    # The data and dictionary are given to the run() method
    def __init__(self, stopval, lambda1, lambda2, lambda2_type="scaled"):

        # parameter check
        if stopval < 0:
            raise ValueError("stopping value is negative")
        if lambda1 < 0:
            raise ValueError("lambda1 is negative")
        if lambda2 < 0:
            raise ValueError("lambda2 is negative")

        self.stopval = stopval
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda2_type = lambda2_type

    def __str__(self):
        #return "UnconstrainedAnalysisPursuit(" + str(self.stopval) + ', ' + str(self.lambda1) + ', ' + str(self.lambda2) + ")"
        return "UAP(" + str(self.stopval) + ', ' + str(self.lambda1) + ', ' + str(self.lambda2) + ")"


    def solve(self, measurements, acqumatrix, operator, realdict=None):

        # Force measurements 2D
        if len(measurements.shape) == 1:
            measurements = np.atleast_2d(measurements)
            if measurements.shape[0] < measurements.shape[1]:
                measurements = np.transpose(measurements)

        numdata = measurements.shape[1]
        signalsize = acqumatrix.shape[1]
        outdata = np.zeros((signalsize, numdata))

        OmegaPinv = None
        P = None
        lambda1 = self.lambda1
        if self.lambda2_type == "value":
            lambda2 = self.lambda2
        elif self.lambda2_type == "scaled":
            gammasize, signalsize = operator.shape
            OmegaPinv = np.linalg.pinv(operator)
            U,S,Vt = np.linalg.svd(OmegaPinv)
            P = Vt[-(gammasize-signalsize):,:]
            #mul = self.nullspace_multiplier / np.linalg.norm(nullspace, 'fro') * nullspace.shape[0] * np.linalg.norm(np.dot(acqumatrix, dictionary), 'fro') / acqumatrix.shape[0]
            lambda2 = self.lambda2 / np.linalg.norm(P, 'fro') * P.shape[0] * np.linalg.norm(np.dot(acqumatrix, OmegaPinv), 'fro') / acqumatrix.shape[0]
        if OmegaPinv is None:
            OmegaPinv = np.linalg.pinv(operator)
        if P is None:
            gammasize, signalsize = operator.shape
            U,S,Vt = np.linalg.svd(OmegaPinv)
            P = Vt[-(gammasize-signalsize):,:]
        for i in range(numdata):
            outdata[:, i] = unconstrained_analysis_pursuit(measurements[:,i], acqumatrix, operator, lambda1, lambda2, OmegaPinv=OmegaPinv, P=P)

        return outdata

def unconstrained_analysis_pursuit(measurements, acqumatrix, operator, lambda1, lambda2, OmegaPinv=None, P=None):

    gammasize, signalsize = operator.shape
    gamma = np.zeros(gammasize)
    Lambdahat = np.arange(gammasize)
    if OmegaPinv is None:
        OmegaPinv = np.linalg.pinv(operator)
    if P is None:
        U,S,Vt = np.linalg.svd(OmegaPinv)
        P = Vt[-(gammasize-signalsize):,:]
    residual = measurements - np.dot(np.dot(acqumatrix, OmegaPinv), gamma)

    while True:  # exit from inside with break

        # Minimization problem
        I_Lambda_k = np.zeros((Lambdahat.size, gammasize))
        I_Lambda_k[range(Lambdahat.size), Lambdahat] = 1
        system_matrix = np.concatenate((np.dot(acqumatrix, OmegaPinv), lambda1 * I_Lambda_k))
        system_matrix = np.concatenate((system_matrix, lambda2 * P))
        y_tilde = np.concatenate((measurements, np.zeros(I_Lambda_k.shape[0] + P.shape[0])))
        # solve
        #gamma = np.linalg.lstsq(system_matrix, y_tilde)[0]
        # Use fast version
        gamma = fast_lstsq(system_matrix, y_tilde)

        # Atom selection
        maxval = np.amax(np.absolute(gamma[Lambdahat]))
        maxrow = Lambdahat[np.argmax(np.absolute(gamma[Lambdahat]))]

        # Scale with GAP or OMP criterion
        #  alternatively: max(maxval, lambda1*maxval)
        if lambda1 > 1:
            maxval = (lambda1**2)*maxval  # lamnda1**2 instead of lambda1 in order to match OMP
        # Exit condition
        if maxval < 1e-6:
            break

        # Remove selected rows
        Lambdahat = np.setdiff1d(Lambdahat, maxrow)

        # Another exit condition
        if Lambdahat.size == 0:
            break

    # Debias (project gamma onto columns of operator)
    gamma = np.dot(np.dot(operator, OmegaPinv), gamma)

    return np.dot(OmegaPinv, gamma)







