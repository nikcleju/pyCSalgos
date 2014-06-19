"""
omp.py

Provides Orthogonal Matching Pursuit (OMP)
"""

# Author: Nicolae Cleju
# License: BSD 3 clause

from base import SparseSolver

try:
    import sklearn.linear_model
    has_sklearn_omp = True
except ImportError, e:
    has_sklearn_omp = False

import omp_sklearn_local

import time
import math
import numpy as np
import scipy

class OrthogonalMatchingPursuit(SparseSolver):
    """
    Attention: compressed sensing problems shouldn't use sklearn's OMP because it assumes that the dictionary
     is normalized, which is not the case with the effective dictionary P*D
     Better use "sparsify_QR" instead.
    """

    # All parameters related to the algorithm itself are given here.
    # The data and dictionary are given to the run() method
    def __init__(self, stopval, algorithm="sklearn"):

        # parameter check
        if stopval < 0:
            raise ValueError("stopping value is negative")

        if stopval < 1:
            self.stopcrit = StopCriterion.TOL
        else:
            self.stopcrit = StopCriterion.FIXED
        self.stopval = stopval
        self.algorithm = algorithm

    def solve(self, data, dictionary):
        return _orthogonal_matching_pursuit(data, dictionary, self.stopval, self.algorithm)


class StopCriterion:
    """
    Stopping criterion type:
        StopCriterion.FIXED:    fixed number of iterations
        StopCriterion.TOL:      until approximation error is below tolerance
    """
    FIXED = 1
    TOL   = 2


def _orthogonal_matching_pursuit(data, dictionary, stopval, algorithm="sklearn"):
    """
    Orthogonal Matching Pursuit algorihtm
    :param data: 2D array containing the data to decompose, columnwise
    :param dictionary: dictionary containing the atoms, columnwise
    :param stopval: stopping criterion
    :param algorithm: what implementation to use
    :return: coefficients

    Available implementations:
      "sklearn" (default)
      "sklearn_local"
      "sparsify_QR"
      "sturm_QR"
    """

    # parameter check
    if stopval < 0:
        raise ValueError("stopping value is negative")
    if stopval > dictionary.shape[0]:
        raise ValueError("stopping value > signal size")
    if stopval > dictionary.shape[1]:
        raise ValueError("stopping value > dictionary size")

    if algorithm == "sklearn" and has_sklearn_omp:
        if stopval < 1:
            # Stop criterion = tolerance
            return sklearn.linear_model.orthogonal_mp(X=dictionary, y=data, tol=stopval)
        else:
            # Stop criterion = No. of nonzero elements
            return sklearn.linear_model.orthogonal_mp(X=dictionary, y=data, n_nonzero_coefs=stopval)

    if (algorithm == "sklearn" and not has_sklearn_omp) or algorithm == "sklearn_local":
        #call local version of sklearn OMP
        if stopval < 1:
            # Stop criterion = tolerance
            return omp_sklearn_local.orthogonal_mp(X=dictionary, y=data, tol=stopval)
        else:
            # Stop criterion = No. of nonzero elements
            return omp_sklearn_local.orthogonal_mp(X=dictionary, y=data, n_nonzero_coefs=stopval)

    if algorithm == "sparsify_QR":
        # call QR-based omp from sparsify package
        ompopts = dict()
        ompopts["nargout"] = 1
        if stopval < 1:
            ompopts["stopCrit"] = "mse"
        else:
            ompopts["stopCrit"] = "M"
        ompopts["stopTol"] = stopval
        if len(data.shape) == 1:
            data = np.atleast_2d(data)
            if data.shape[0] < data.shape[1]:
                data = np.transpose(data)
        coef = np.zeros((dictionary.shape[1], data.shape[1]))
        for i in range(data.shape[1]):
            coef[:,i] = omp_sparsify_greed_omp_qr(data[:,i], dictionary, dictionary.shape[1], ompopts)
        return np.squeeze(coef)

    if algorithm == "sturm_QR":
        # call QR-based OMP by Bob Sturm
        if len(data.shape) == 1:
            data = np.atleast_2d(data)
            if data.shape[0] < data.shape[1]:
                data = np.transpose(data)
        coef = np.zeros((dictionary.shape[1], data.shape[1]))
        for i in range(data.shape[1]):
            if stopval < 1:
                coef[:,i], support = omp_sturm_omp_qr(data[:,i], dictionary, np.dot(dictionary.T, dictionary), data.shape[0], stopval)
            else:
                coef[:,i], support = omp_sturm_omp_qr(data[:,i], dictionary, np.dot(dictionary.T, dictionary), stopval, 0)
        return np.squeeze(coef)

    raise ValueError("Algorithm '%s' does not exist", algorithm)


def omp_sparsify_greed_omp_qr(x,A,m,opts=[]):
    # greed_omp_qr: Orthogonal Matching Pursuit algorithm based on QR
    # factorisation
    # Nic: translated to Python on 19.10.2011. Original Matlab Code by Thomas Blumensath

    if x.ndim != 1:
        print 'x must be a vector.'
        return
    n = x.size

    sigsize     = np.vdot(x,x)/n;
    initial_given = 0;
    err_mse     = np.array([]);
    iter_time   = np.array([]);
    STOPCRIT    = 'M';
    STOPTOL     = math.ceil(n/4.0);
    MAXITER     = n;
    verbose     = False;
    s_initial   = np.zeros(m);

    if verbose:
        print 'Initialising...'
    #end

    ###########################################################################
    #                           Output variables
    ###########################################################################
    if 'nargout' in opts:
        if opts['nargout'] == 3:
            comp_err  = True
            comp_time = True
        elif opts['nargout'] == 2:
            comp_err  = True
            comp_time = False
        elif opts['nargout'] == 1:
            comp_err  = False
            comp_time = False
        elif opts['nargout'] == 0:
            print 'Please assign output variable.'
            return
        else:
            print 'Too many output arguments specified'
            return
    else:
        # If not given, make default nargout = 3
        #  and add nargout to options
        opts['nargout'] = 3
        comp_err  = True
        comp_time = True

    ###########################################################################
    #                       Look through options
    ###########################################################################
    if 'stopCrit' in opts:
        STOPCRIT = opts['stopCrit']
    if 'stopTol' in opts:
        if hasattr(opts['stopTol'], '__int__'):  # check if numeric
            STOPTOL = opts['stopTol']
        else:
            raise TypeError('stopTol must be number. Exiting.')
    if 'P_trans' in opts:
        if hasattr(opts['P_trans'], '__call__'):  # check if function handle
            Pt = opts['P_trans']
        else:
            raise TypeError('P_trans must be function _handle. Exiting.')
    if 'maxIter' in opts:
        if hasattr(opts['maxIter'], '__int__'):  # check if numeric
            MAXITER = opts['maxIter']
        else:
            raise TypeError('maxIter must be a number. Exiting.')
    if 'verbose' in opts:
        # TODO: Should check here if is logical
        verbose = opts['verbose']
    if 'start_val' in opts:
        # TODO: Should check here if is numeric
        if opts['start_val'].size == m:
            s_initial = opts['start_val']
            initial_given = 1
        else:
            raise ValueError('start_val must be a vector of length m. Exiting.')
    # Don't exit if unknown option is given, simply ignore it

    if STOPCRIT == 'M':
        maxM = STOPTOL
    else:
        maxM = MAXITER

    if opts['nargout'] >= 2:
        err_mse = np.zeros(maxM)
    if opts['nargout'] == 3:
        iter_time = np.zeros(maxM)

    ###########################################################################
    #                        Make P and Pt functions
    ###########################################################################
    if hasattr(A, '__call__'):
        if hasattr(Pt, '__call__'):
            P = A
        else:
            raise TypeError('If P is a function handle, Pt also needs to be a function handle.')
    else:
        # TODO: should check here if A is matrix
        P  = lambda z: np.dot(A,z)
        Pt = lambda z: np.dot(A.T,z)

    ###########################################################################
    #                 Random Check to see if dictionary is normalised
    ###########################################################################
    # Don't do this.

    ###########################################################################
    #              Check if we have enough memory and initialise
    ###########################################################################
    try:
        Q = np.zeros((n,maxM))
    except:
        print 'Variable size is too large. Please try greed_omp_chol algorithm or reduce MAXITER.'
        raise
    try:
        R = np.zeros((maxM, maxM))
    except:
        print 'Variable size is too large. Please try greed_omp_chol algorithm or reduce MAXITER.'
        raise

    ###########################################################################
    #                        Do we start from zero or not?
    ###########################################################################
    if initial_given == 1:
        IN = np.nonzero(s_initial)[0].tolist()
        if IN.size > 0:
            Residual = x - P(s_initial)
            lengthIN = IN.size
            z = np.array([])
            for k in np.arange(IN.size):
                # Extract new element
                mask = np.zeros(m)
                mask[IN[k]] = 1
                new_element = P(mask)

                # Orthogonalise new element
                if k-1 >= 0:
                    qP = np.dot(Q[:,0:k].T , new_element)
                    q = new_element - np.dot(Q[:,0:k] , qP)

                    nq = np.linalg.norm(q)
                    q = q / nq
                    # Update QR factorisation
                    R[0:k,k] = qP
                    R[k,k] = nq
                    Q[:,k] = q
                else:
                    q = new_element

                    nq = np.linalg.norm(q)
                    q = q / nq
                    # Update QR factorisation
                    R[k,k] = nq
                    Q[:,k] = q

                z[k] = np.dot(q.T , x)
            s        = s_initial.copy()
            Residual = x - np.dot(Q[:,k] , z)
            oldERR   = np.vdot(Residual , Residual) / n;
        else:
            IN          = np.array([], dtype = int).tolist()
            Residual    = x.copy()
            s           = s_initial.copy()
            sigsize     = np.vdot(x , x) / n
            oldERR      = sigsize
            k = 0
            z = []

    else:
        IN          = np.array([], dtype = int).tolist()
        Residual    = x.copy()
        s           = s_initial.copy()
        sigsize     = np.vdot(x , x) / n
        oldERR      = sigsize
        k = 0
        z = []

    ###########################################################################
    #                        Main algorithm
    ###########################################################################
    if verbose:
        print 'Main iterations...'
    tic = time.time()
    t = 0
    DR = Pt(Residual)
    done = 0
    iter = 1

    while not done:

        # Select new element
        DR[IN]=0
        I = np.abs(DR).argmax()
        IN.append(I)

        # Extract new element
        mask = np.zeros(m)
        mask[IN[k]] = 1
        new_element = P(mask)

        # Orthogonalise new element
        if k-1 >= 0:
            qP = np.dot(Q[:,0:k].T , new_element)
            q = new_element - np.dot(Q[:,0:k] , qP)

            nq = np.linalg.norm(q)
            q = q/nq
            # Update QR factorisation
            R[0:k,k] = qP
            R[k,k] = nq
            Q[:,k] = q
        else:
            q = new_element

            nq = np.linalg.norm(q)
            q = q/nq
            # Update QR factorisation
            R[k,k] = nq
            Q[:,k] = q

        z.append(np.vdot(q , x))

        # New residual
        Residual = Residual - q * (z[k])
        DR = Pt(Residual)

        ERR = np.vdot(Residual , Residual) / n
        if comp_err:
            err_mse[iter-1] = ERR

        if comp_time:
            iter_time[iter-1] = time.time() - tic

        ###########################################################################
        #                        Are we done yet?
        ###########################################################################
        if STOPCRIT == 'M':
            if iter >= STOPTOL:
                done = 1
            elif verbose and time.time() - t > 10.0/1000: # time() returns sec
                print 'Iteration '+iter+'. --- '+(STOPTOL-iter)+' iterations to go'
                t = time.time()
        elif STOPCRIT =='mse':
            if comp_err:
                if err_mse[iter-1] < STOPTOL:
                    done = 1
                elif verbose and time.time() - t > 10.0/1000: # time() returns sec
                    print 'Iteration '+iter+'. --- '+err_mse[iter-1]+' mse'
                    t = time.time()
            else:
                if ERR < STOPTOL:
                    done = 1
                elif verbose and time.time() - t > 10.0/1000: # time() returns sec
                    print 'Iteration '+iter+'. --- '+ERR+' mse'
                    t = time.time()
        elif STOPCRIT == 'mse_change' and iter >=2:
            if comp_err and iter >=2:
                if ((err_mse[iter-2] - err_mse[iter-1])/sigsize < STOPTOL):
                    done = 1
                elif verbose and time.time() - t > 10.0/1000: # time() returns sec
                    print 'Iteration '+iter+'. --- '+((err_mse[iter-2]-err_mse[iter-1])/sigsize)+' mse change'
                    t = time.time()
            else:
                if ((oldERR - ERR)/sigsize < STOPTOL):
                    done = 1
                elif verbose and time.time() - t > 10.0/1000: # time() returns sec
                    print 'Iteration '+iter+'. --- '+((oldERR - ERR)/sigsize)+' mse change'
                    t = time.time()
        elif STOPCRIT == 'corr':
            if np.abs(DR).max() < STOPTOL:
                done = 1
            elif verbose and time.time() - t > 10.0/1000: # time() returns sec
                print 'Iteration '+iter+'. --- '+(np.abs(DR).max())+' corr'
                t = time.time()

        # Also stop if residual gets too small or maxIter reached
        if comp_err:
            if err_mse[iter-1] < 1e-14:
                done = 1
                if verbose:
                    print 'Stopping. Exact signal representation found!'
        else:
            if iter > 1:
                if ERR < 1e-14:
                    done = 1
                    if verbose:
                        print 'Stopping. Exact signal representation found!'

        if iter >= MAXITER:
            done = 1
            if verbose:
                print 'Stopping. Maximum number of iterations reached!'

        ###########################################################################
        #                    If not done, take another round
        ###########################################################################
        if not done:
            iter = iter + 1
            oldERR = ERR

        # Moved here from front, since we are 0-based
        k = k + 1

    ###########################################################################
    #            Now we can solve for s by back-substitution
    ###########################################################################
    s[IN] = scipy.linalg.solve(R[0:k,0:k] , np.array(z[0:k]))

    ###########################################################################
    #                  Only return as many elements as iterations
    ###########################################################################
    if opts['nargout'] >= 2:
        err_mse = err_mse[0:iter-1]
    if opts['nargout'] == 3:
        iter_time = iter_time[0:iter-1]
    if verbose:
        print 'Done'

    # Return
    if opts['nargout'] == 1:
        return s
    elif opts['nargout'] == 2:
        return s, err_mse
    elif opts['nargout'] == 3:
        return s, err_mse, iter_time


def omp_sturm_omp_qr(x, dict, D, natom, tolerance):
    """ Recover x using QR implementation of OMP

   Parameter
   ---------
   x: measurements
   dict: dictionary
   D: Gramian of dictionary
   natom: iterations
   tolerance: error tolerance

   Return
   ------
   x_hat : estimate of x
   gamma : indices where non-zero

   For more information, see http://media.aau.dk/null_space_pursuits/2011/10/efficient-omp.html
   """
    msize, dictsize = dict.shape
    normr2 = np.vdot(x,x)
    normtol2 = tolerance*normr2
    R = np.zeros((natom,natom))
    Q = np.zeros((msize,natom))
    gamma = []

    # find initial projections
    origprojections = np.dot(x.T,dict)
    origprojectionsT = origprojections.T
    projections = origprojections.copy();

    k = 0
    while (normr2 > normtol2) and (k < natom):
        # find index of maximum magnitude projection
        newgam = np.argmax(np.abs(projections ** 2))
        gamma.append(newgam)
        # update QR factorization, projections, and residual energy
        if k == 0:
            R[0,0] = 1
            Q[:,0] = dict[:,newgam].copy()
            # update projections
            QtempQtempT = np.outer(Q[:,0],Q[:,0])
            projections -= np.dot(x.T, np.dot(QtempQtempT,dict))
            # update residual energy
            normr2 -= np.vdot(x, np.dot(QtempQtempT,x))
        else:
            w = scipy.linalg.solve_triangular(R[0:k,0:k],D[gamma[0:k],newgam],trans=1)
            R[k,k] = np.sqrt(1-np.vdot(w,w))
            R[0:k,k] = w.copy()
            Q[:,k] = (dict[:,newgam] - np.dot(QtempQtempT,dict[:,newgam]))/R[k,k]
            QkQkT = np.outer(Q[:,k],Q[:,k])
            xTQkQkT = np.dot(x.T,QkQkT)
            QtempQtempT += QkQkT
            # update projections
            projections -= np.dot(xTQkQkT,dict)
            # update residual energy
            normr2 -= np.dot(xTQkQkT,x)

        k += 1

    # build solution
    tempR = R[0:k,0:k]
    w = scipy.linalg.solve_triangular(tempR,origprojectionsT[gamma[0:k]],trans=1)
    #x_hat = np.zeros((dictsize,1))
    x_hat = np.zeros((dictsize))
    x_hat[gamma[0:k]] = scipy.linalg.solve_triangular(tempR,w)

    return x_hat, gamma
