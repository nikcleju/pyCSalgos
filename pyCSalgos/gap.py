"""
gap.py

Provides Greedy Analysis Pursuit(GAP) for analysis-based recovery
"""

# Author: Nicolae Cleju (translated to Python)
# License: BSD 3 clause

import math
import scipy

import numpy as np

from base import AnalysisSparseSolver, ERCcheckMixin


class GreedyAnalysisPursuit(ERCcheckMixin, AnalysisSparseSolver):

    # All parameters related to the algorithm itself are given here.
    # The data and dictionary are given to the run() method
    def __init__(self, stopval):

        # parameter check
        if stopval < 0:
            raise ValueError("stopping value is negative")

        self.stopval = stopval

    def __str__(self):
        return "GreedyAnalysisPursuit("+str(self.stopval) + ")"


    def solve(self, measurements, acqumatrix, operator):

        # Force measurements 2D
        if len(measurements.shape) == 1:
            measurements = np.atleast_2d(measurements)
            if measurements.shape[0] < measurements.shape[1]:
                measurements = np.transpose(measurements)

        numdata = measurements.shape[1]
        signalsize = acqumatrix.shape[1]
        outdata = np.zeros((signalsize, numdata))

        gapparams = {"num_iteration" : 1000,
                     "greedy_level" : 0.9,
                     "stopping_coefficient_size" : 1e-4,
                     "l2solver" : 'pseudoinverse',
                     "noise_level": self.stopval}
        for i in range(numdata):
            outdata[:, i] = greedy_analysis_pursuit(measurements[:,i], acqumatrix, acqumatrix.T, operator, operator.T,
                                                    gapparams, np.zeros(operator.shape[1]))[0]
        return np.squeeze(outdata)

    def checkERC(self, acqumatrix, dictoper, support):

        # Should normalize here the operator or not?

        [operatorsize, signalsize] = dictoper.shape
        U,S,Vt = np.linalg.svd(acqumatrix)
        N = Vt[-(operatorsize-signalsize):, :]

        k = support.shape[0]
        l = operatorsize - k
        num_data = support.shape[1]
        results = np.zeros(num_data, dtype=bool)

        for i in range(num_data):
            LambdaC = support[:,i]
            Lambda = np.setdiff1d(range(operatorsize), LambdaC)
            A = np.dot(
                np.linalg.pinv(np.dot(N, dictoper[Lambda,:].T)),
                np.dot(N, dictoper[LambdaC,:].T))
            assert(A.shape == (l, k))

            linf = np.max(np.sum(np.abs(A),1)) # L_infty,infty matrix norm
            if linf < 1:
                results[i] = True
            else:
                results[i] = False
        return results



def ArgminOperL2Constrained(y, M, MH, Omega, OmegaH, Lambdahat, xinit, ilagmult, params):

    # This function aims to compute
    #    xhat = argmin || Omega(Lambdahat, :) * x ||_2   subject to  || y - M*x ||_2 <= epsilon.
    # arepr is the analysis representation corresponding to Lambdahat, i.e.,
    #    arepr = Omega(Lambdahat, :) * xhat.
    # The function also returns the lagrange multiplier in the process used to compute xhat.
    #
    # Inputs:
    #    y : observation/measurements of an unknown vector x0. It is equal to M*x0 + noise.
    #    M : Measurement matrix
    #    MH : M', the conjugate transpose of M
    #    Omega : analysis operator
    #    OmegaH : Omega', the conjugate transpose of Omega. Also, synthesis operator.
    #    Lambdahat : an index set indicating some rows of Omega.
    #    xinit : initial estimate that will be used for the conjugate gradient algorithm.
    #    ilagmult : initial lagrange multiplier to be used in
    #    params : parameters
    #        params.noise_level : this corresponds to epsilon above.
    #        params.max_inner_iteration : `maximum' number of iterations in conjugate gradient method.
    #        params.l2_accurary : the l2 accuracy parameter used in conjugate gradient method
    #        params.l2solver : if the value is 'pseudoinverse', then direct matrix computation (not conjugate gradient method) is used. Otherwise, conjugate gradient method is used.

    d = xinit.size
    lagmultmax = 1e5
    lagmultmin = 1e-4
    lagmultfactor = 2.0
    accuracy_adjustment_exponent = 4/5.
    lagmult = max(min(ilagmult, lagmultmax), lagmultmin)
    was_infeasible = 0
    was_feasible = 0

    #######################################################################
    ## Computation done using direct matrix computation from matlab. (no conjugate gradient method.)
    #######################################################################
    if params['l2solver'] == 'pseudoinverse':
        if 1:
            while True:
                alpha = math.sqrt(lagmult)

                # Build augmented matrix and measurements vector
                Omega_tilde = np.concatenate((M, alpha*Omega[Lambdahat,:]))
                y_tilde = np.concatenate((y, np.zeros(Lambdahat.size)))

                # Solve least-squares problem
                xhat = np.linalg.lstsq(Omega_tilde, y_tilde)[0]

                # Check tolerance below required, and adjust Lagr multiplier accordingly
                temp = np.linalg.norm(y - np.dot(M,xhat), 2)
                if temp <= params['noise_level']:
                    was_feasible = True
                    if was_infeasible:
                        break
                    else:
                        lagmult = lagmult*lagmultfactor
                elif temp > params['noise_level']:
                    was_infeasible = True
                    if was_feasible:
                        xhat = xprev.copy()
                        break
                    lagmult = lagmult/lagmultfactor
                if lagmult < lagmultmin or lagmult > lagmultmax:
                    break
                xprev = xhat.copy()

            arepr = np.dot(Omega[Lambdahat, :], xhat)
            return xhat,arepr,lagmult


    ########################################################################
    ## Computation using conjugate gradient method.
    ########################################################################
    if hasattr(MH, '__call__'):
        b = MH(y)
    else:
        b = np.dot(MH, y)

    norm_b = np.linalg.norm(b, 2)
    xhat = xinit.copy()
    xprev = xinit.copy()
    residual = TheHermitianMatrix(xhat, M, MH, Omega, OmegaH, Lambdahat, lagmult) - b
    direction = -residual
    iter = 0

    while iter < params.max_inner_iteration:
        iter = iter + 1;
        alpha = np.linalg.norm(residual,2)**2 / np.dot(direction.T, TheHermitianMatrix(direction, M, MH, Omega, OmegaH, Lambdahat, lagmult));
        xhat = xhat + alpha*direction;
        prev_residual = residual.copy();
        residual = TheHermitianMatrix(xhat, M, MH, Omega, OmegaH, Lambdahat, lagmult) - b;
        beta = np.linalg.norm(residual,2)**2 / np.linalg.norm(prev_residual,2)**2;
        direction = -residual + beta*direction;

        if np.linalg.norm(residual,2)/norm_b < params['l2_accuracy']*(lagmult**(accuracy_adjustment_exponent)) or iter == params['max_inner_iteration']:
            if hasattr(M, '__call__'):
                temp = np.linalg.norm(y-M(xhat), 2);
            else:
                temp = np.linalg.norm(y-np.dot(M,xhat), 2);

            #if strcmp(class(Omega), 'function_handle')
            if hasattr(Omega, '__call__'):
                u = Omega(xhat);
                u = math.sqrt(lagmult)*np.linalg.norm(u(Lambdahat), 2);
            else:
                u = math.sqrt(lagmult)*np.linalg.norm(Omega[Lambdahat,:]*xhat, 2);


            if temp <= params['noise_level']:
                was_feasible = True;
                if was_infeasible:
                    break;
                else:
                    lagmult = lagmultfactor*lagmult;
                    residual = TheHermitianMatrix(xhat, M, MH, Omega, OmegaH, Lambdahat, lagmult) - b;
                    direction = -residual;
                    iter = 0;
            elif temp > params['noise_level']:
                lagmult = lagmult/lagmultfactor;
                if was_feasible:
                    xhat = xprev.copy();
                    break;
                was_infeasible = True;
                residual = TheHermitianMatrix(xhat, M, MH, Omega, OmegaH, Lambdahat, lagmult) - b;
                direction = -residual;
                iter = 0;
            if lagmult > lagmultmax or lagmult < lagmultmin:
                break;
            xprev = xhat.copy();

    print 'fidelity_error=',temp

    ##
    # Compute analysis representation for xhat
    ##
    if hasattr(Omega, '__call__'):
        temp = Omega(xhat);
        arepr = temp(Lambdahat);
    else:    ## here Omega is assumed to be a matrix
        arepr = np.dot(Omega[Lambdahat, :], xhat);

    return xhat,arepr,lagmult


##
# This function computes (M'*M + lm*Omega(L,:)'*Omega(L,:)) * x.
##
def TheHermitianMatrix(x, M, MH, Omega, OmegaH, L, lm):
    if hasattr(M, '__call__'):
        w = MH(M(x));
    else:    ## M and MH are matrices
        w = np.dot(np.dot(MH, M), x);

    if hasattr(Omega, '__call__'):
        v = Omega(x);
        vt = np.zeros(v.size);
        vt[L] = v[L].copy();
        w = w + lm*OmegaH(vt);
    else:    ## Omega is assumed to be a matrix and OmegaH is its conjugate transpose
        w = w + lm*np.dot(np.dot(OmegaH[:, L],Omega[L, :]),x);

    return w

def greedy_analysis_pursuit(y, M, MH, Omega, OmegaH, params, xinit):
    ##
    # [xhat, Lambdahat] = GAP(y, M, MH, Omega, OmegaH, params, xinit)
    #
    # Greedy Analysis Pursuit Algorithm
    # This aims to find an approximate (sometimes exact) solution of
    #    xhat = argmin || Omega * x ||_0   subject to   || y - M * x ||_2 <= epsilon.
    #
    # Outputs:
    #   xhat : estimate of the target cosparse vector x0.
    #   Lambdahat : estimate of the cosupport of x0.
    #
    # Inputs:
    #   y : observation/measurement vector of a target cosparse solution x0,
    #       given by relation  y = M * x0 + noise.
    #   M : measurement matrix. This should be given either as a matrix or as a function handle
    #       which implements linear transformation.
    #   MH : conjugate transpose of M.
    #   Omega : analysis operator. Like M, this should be given either as a matrix or as a function
    #           handle which implements linear transformation.
    #   OmegaH : conjugate transpose of OmegaH.
    #   params : parameters that govern the behavior of the algorithm (mostly).
    #      params.num_iteration : GAP performs this number of iterations.
    #      params.greedy_level : determines how many rows of Omega GAP eliminates at each iteration.
    #                            if the value is < 1, then the rows to be eliminated are determined by
    #                                j : |omega_j * xhat| > greedy_level * max_i |omega_i * xhat|.
    #                            if the value is >= 1, then greedy_level is the number of rows to be
    #                            eliminated at each iteration.
    #      params.stopping_coefficient_size : when the maximum analysis coefficient is smaller than
    #                                         this, GAP terminates.
    #      params.l2solver : legitimate values are 'pseudoinverse' or 'cg'. determines which method
    #                        is used to compute
    #                        argmin || Omega_Lambdahat * x ||_2   subject to  || y - M * x ||_2 <= epsilon.
    #      params.l2_accuracy : when l2solver is 'cg', this determines how accurately the above
    #                           problem is solved.
    #      params.noise_level : this corresponds to epsilon above.
    #   xinit : initial estimate of x0 that GAP will start with. can be zeros(d, 1).
    #
    # Examples:
    #
    # Not particularly interesting:
    # >> d = 100; p = 110; m = 60;
    # >> M = randn(m, d);
    # >> Omega = randn(p, d);
    # >> y = M * x0 + noise;
    # >> params.num_iteration = 40;
    # >> params.greedy_level = 0.9;
    # >> params.stopping_coefficient_size = 1e-4;
    # >> params.l2solver = 'pseudoinverse';
    # >> [xhat, Lambdahat] = GAP(y, M, M', Omega, Omega', params, zeros(d, 1));
    #
    # Assuming that FourierSampling.m, FourierSamplingH.m, FDAnalysis.m, etc. exist:
    # >> n = 128;
    # >> M = @(t) FourierSampling(t, n);
    # >> MH = @(u) FourierSamplingH(u, n);
    # >> Omega = @(t) FDAnalysis(t, n);
    # >> OmegaH = @(u) FDSynthesis(t, n);
    # >> params.num_iteration = 1000;
    # >> params.greedy_level = 50;
    # >> params.stopping_coefficient_size = 1e-5;
    # >> params.l2solver = 'cg';   # in fact, 'pseudoinverse' does not even make sense.
    # >> [xhat, Lambdahat] = GAP(y, M, MH, Omega, OmegaH, params, zeros(d, 1));
    #
    # Above: FourierSampling and FourierSamplingH are conjugate transpose of each other.
    #        FDAnalysis and FDSynthesis are conjugate transpose of each other.
    #        These routines are problem specific and need to be implemented by the user.

    d = xinit.size

    y = np.squeeze(y) # to double-check

    if hasattr(Omega, '__call__'):
        p = Omega(np.zeros((d,1)))
    else:
        p = Omega.shape[0]


    iter = 0
    lagmult = 1e-4
    Lambdahat = np.arange(p)

    while iter < params["num_iteration"]:
        iter = iter + 1
        xhat,analysis_repr,lagmult = ArgminOperL2Constrained(y, M, MH, Omega, OmegaH, Lambdahat, xinit, lagmult, params)
        to_be_removed,maxcoef = FindRowsToRemove(analysis_repr, params["greedy_level"])
        if check_stopping_criteria(xhat, xinit, maxcoef, lagmult, Lambdahat, params):
            break

        xinit = xhat.copy()
        Lambdahat = np.delete(Lambdahat.squeeze(),to_be_removed)

        # Added by Nic: if Lambdahat is empty here, return
        if Lambdahat.size == 0:
            break

    return xhat,Lambdahat

def FindRowsToRemove(analysis_repr, greedy_level):

    abscoef = np.abs(analysis_repr)
    n = abscoef.size
    maxcoef = abscoef.max()
    if greedy_level >= 1:
        qq = scipy.stats.mstats.mquantiles(abscoef, 1.0-greedy_level/n)
    else:
        qq = maxcoef*greedy_level

    # [0] needed because nonzero() returns a tuple of arrays!
    to_be_removed = np.nonzero(abscoef >= qq)[0]
    return to_be_removed,maxcoef

def check_stopping_criteria(xhat, xinit, maxcoef, lagmult, Lambdahat, params):

    if ('stopping_coefficient_size' in params) and maxcoef < params['stopping_coefficient_size']:
        return 1

    if ('stopping_lagrange_multiplier_size' in params) and lagmult > params['stopping_lagrange_multiplier_size']:
        return 1

    if ('stopping_relative_solution_change' in params) and np.linalg.norm(xhat-xinit)/np.linalg.norm(xhat) < params['stopping_relative_solution_change']:
        return 1

    if ('stopping_cosparsity' in params) and Lambdahat.size < params['stopping_cosparsity']:
        return 1

    return 0