"""
iht.py

Provides Iterative Hard Thresholding (Accelerated IHT)
"""

# Author: Nicolae Cleju
# License: BSD 3 clause

import math

import numpy as np

from .base import SparseSolver

import warnings


class IterativeHardThresholding(SparseSolver):
    """
    Iterative Hard Thresholding
    """

    def __init__(self, stoptol, deltatol=1e-10, sparsity="half", maxiter=70000, debias=True):

        # parameter check
        if stoptol < 0:
            raise ValueError("stopping tolerance is negative")
        if maxiter <= 0:
            raise ValueError("number of iterations is not positive")

        self.stoptol = stoptol
        self.deltatol = deltatol
        self.maxiter = maxiter
        self.sparsity = sparsity
        self.debias = debias

    def __str__(self):
        return "IHT (" + str(self.stoptol) + " | " + str(self.maxiter) + " | " +  str(self.sparsity) + ")"

    def solve(self, data_orig, dictionary_orig, realdict=None):

        # DEBUG:
        norm = np.linalg.norm(dictionary_orig, 2)
        # use more than the l2 norm here, to ensure stability => use frobenius norm
        #norm = np.linalg.norm(dictionary_orig, 'fro')
        #norm = 1. / np.sqrt(data_orig.shape[0])
        dictionary = dictionary_orig.copy() / norm
        data = data_orig.copy() / norm

        # Force data 2D
        if len(data.shape) == 1:
            data = np.atleast_2d(data)
            if data.shape[0] < data.shape[1]:
                data = np.transpose(data)

        N = dictionary.shape[1]
        Ndata = data.shape[1]
        coef = np.zeros((N, Ndata))

        for i in range(Ndata):
            if self.sparsity == "real":
                if realdict is not None:
                    if 'support' in realdict:
                        M = realdict['support'].shape[0]
                    elif 'gamma' in realdict:
                        M = np.nonzero(realdict['gamma'][:, i]).size()
                    else:
                        raise ValueError('IHT Error: sparsity set to "real" but no real support or '
                                         'sparse vector given')
                else:
                    raise ValueError('IHT Error: sparsity set to "real" but no real dictionary given')
            elif self.sparsity == "half":
                M = int(round(data.shape[0] / 2))
            else:
                M = self.sparsity  #TODO check type

            coef[:, i] = _iht(dictionary, data[:, i], sparsity=M, errortol=self.stoptol, deltatol=self.deltatol, maxiter=self.maxiter)

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


def _iht(dictionary, measurements, sparsity=None, deltatol=1e-10, errortol=0, maxiter=500, algorithm="accelerated"):
    # x   Observation vector to be decomposed
    #               P   Either:
    #                       1) An nxm matrix (n must be dimension of x)
    #                       2) A function handle (type "help function_format"
    #                                                   %                          for more information)
    #                          Also requires specification of P_trans option.
    #                       3) An object handle (type "help object_format" for
    #                          more information)
    #               m   length of s
    #               M   non-zero elements to keep in each iteration
    #               tol = threshold for modified convergence criterion, not error-based! (Nic)
    #
    #   Possible additional options:
    #   (specify as many as you want using 'option_name','option_value' pairs)
    #   See below for explanation of options:

    n = measurements.shape[0]
    m = dictionary.shape[1]

    M = sparsity  # renaming
    if M is None:
        M = int(round(n / 2))  # Nic: just like in my OptimProj paper. No of non-zero elements to keep is at most m/2

    sigsize = np.dot(measurements.T, measurements) / n
    oldERR = sigsize
    err_mse = []
    iter_time = []
    s_initial = np.zeros(m)
    MU = 0
    acceleration = -1 #DEBUG: disabled (disable=-1)

    Count = 0

    # Make P and Pt functions
    if hasattr(dictionary, '__call__'):
        raise RuntimeError("Large-scale not implemented yet")
    else:
        #P =@(z) A*z;
        #Pt =@(z) A'*z;
        P = lambda z: np.dot(dictionary, z)
        Pt = lambda z: np.dot(dictionary.T, z)

    s_initial = np.zeros(m)
    Residual = measurements.copy()
    s = s_initial.copy()
    Ps = np.zeros(n)
    oldERR = sigsize


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                 Random Check to see if dictionary norm is below 1
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    x_test = np.random.randn(m)
    x_test = x_test / np.linalg.norm(x_test, 2)
    nP = np.linalg.norm(P(x_test), 2)
    if np.abs(MU * nP) > 1:
        warnings.warn('WARNING! Algorithm likely to become unstable.', RuntimeWarning)
        warnings.warn('Use smaller step-size or || P ||_2 < 1.', RuntimeWarning)


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                        Main algorithm
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    t = 0
    done = 0
    iter = 1
    #min_mu = 100000
    #max_mu = 0
    max_mu = 100000
    min_mu = 0

    while not done:
        Count = Count + 1

        if MU == 0:

            #Calculate optimal step size and do line search
            if Count > 1 and acceleration == 0:
                s_very_old = s_old.copy()

            s_old = s.copy()

            #IND                 =   s~=0
            IND = (s != 0)
            d = Pt(Residual)
            # If the current vector is zero, we take the largest elements in d
            if np.sum(IND) == 0:
                #[dsort sortdind]    =   sort(abs(d),'descend');
                sortdind = np.argsort(np.abs(d))[::-1]
                dsort = np.abs(d)[sortdind]
                #IND(sortdind(1:M))  =   1;
                IND[sortdind[:M]] = 1

            id = IND * d
            Pd = P(id)

            if 'mu' in locals(): # if mu exists
                mu_old = mu
            mu = np.dot(id.T, id) / np.dot(Pd.T, Pd)

            if not np.isfinite(mu):
                warnings.warn('IHT: mu is not finite, reverting to old value', RuntimeWarning)
                mu = mu_old #DEBUG

            #max_mu              =   max([mu, max_mu])
            #min_mu              =   min([mu, min_mu])
            #mu                  =   min_mu
            mu = min([mu, max_mu])
            mu = max([mu, min_mu])

            s = s_old + mu * d
            if not all(np.isfinite(s)):
                warnings.warn('IHT: s is not finite', RuntimeWarning)

            #[ssort sortind]     =   sort(abs(s),'descend');
            sortind = np.argsort(np.abs(s))[::-1]
            ssort = np.abs(s)[sortind]
            #s(sortind(M+1:end)) =   0;
            s[sortind[M:]] = 0
            if Count > 1 and acceleration == 0:
                very_old_Ps = old_Ps.copy()

            old_Ps = Ps.copy()
            Ps = P(s)
            Residual = measurements - Ps

            if Count > 2 and acceleration == 0:
                # 1st over-relaxation
                Dif = (Ps - old_Ps)
                #a1                  = Dif'*Residual/ (Dif'*Dif);
                #if(np.linalg.norm(Dif) < 1e-10):
                #    raise RuntimeError('')
                a1 = np.dot(Dif.T, Residual) / np.dot(Dif.T, Dif)
                z1 = s + a1 * (s - s_old)
                Pz1 = (1 + a1) * Ps - a1 * old_Ps
                Residual_z1 = measurements - Pz1

                # 2nd over-relaxation
                Dif = (Pz1 - very_old_Ps)
                a2 = np.dot(Dif.T, Residual_z1) / np.dot(Dif.T, Dif)
                z2 = z1 + a2 * (z1 - s_very_old)

                # Threshold z2
                #[z2sort sortind]     =   sort(abs(z2),'descend');
                sortind = np.argsort(np.abs(z2))[::-1]
                z2sort = np.abs(z2)[sortind]
                #z2(sortind(M+1:end)) =   0;
                z2[sortind[M:]] = 0
                Pz2 = P(z2)
                Residual_z2 = measurements - Pz2

                # Decide if z2 is any good

                if np.dot(Residual_z2.T, Residual_z2) / np.dot(Residual.T, Residual) < 1:
                    s = z2.copy()
                    Residual = Residual_z2.copy()
                    Ps = Pz2.copy()

            if acceleration > 0:
                #[s Residual] =MySubsetCG(x,s,P,Pt,find(s~=0),1e-9,0,CGSteps);
                #Ps           = P(s);
                raise RuntimeError('Large-scale not implemented yet')


            # Calculate step-size requirement
            #omega               =   (norm(s-s_old)/norm(Ps-old_Ps))^2;
            omega = (np.linalg.norm(s - s_old, 2) / np.linalg.norm(Ps - old_Ps, 2)) ** 2

            # As long as the support changes and mu > omega, we decrease mu
            #while mu >= 1.5*omega && sum(xor(IND,s~=0))~=0 && sum(IND)~=0
            while (mu >= 1.5 * omega) and (np.sum(np.logical_xor(IND, (s != 0))) is not 0) and (np.sum(IND) is not 0):
                #display(['decreasing mu'])

                # We use a simple line search, halving mu in each step
                mu = mu / 2
                s = s_old + mu * d
                #[ssort sortind]     =   sort(abs(s),'descend');
                sortind = np.argsort(np.abs(s))[::-1]
                ssort = np.abs(s)[sortind]
                #s(sortind(M+1:end)) =   0;
                s[sortind[M:]] = 0
                Ps = P(s)
                #Calculate optimal step size and do line search
                Residual = measurements - Ps
                if Count > 2 and acceleration == 0:
                    # 1st over-relaxation
                    Dif = (Ps - old_Ps)
                    #if(np.linalg.norm(Dif) < 1e-10):
                    #    raise RuntimeError('')
                    a1 = np.dot(Dif.T, Residual) / np.dot(Dif.T, Dif)
                    z1 = s + a1 * (s - s_old)
                    Pz1 = (1 + a1) * Ps - a1 * old_Ps
                    Residual_z1 = measurements - Pz1

                    # 2nd over-relaxation
                    Dif = (Pz1 - very_old_Ps)
                    a2 = np.dot(Dif.T, Residual_z1) / np.dot(Dif.T, Dif)
                    z2 = z1 + a2 * (z1 - s_very_old)

                    # Threshold z2
                    #[z2sort sortind]     =   sort(abs(z2),'descend');
                    sortind = np.argsort(np.abs(z2))[::-1]
                    z2sort = np.abs(z2)[sortind]
                    #z2(sortind(M+1:end)) =   0;
                    z2[sortind[M:]] = 0
                    Pz2 = P(z2)
                    Residual_z2 = measurements - Pz2

                    # Decide if z2 is any good

                    if np.dot(Residual_z2.T, Residual_z2) / np.dot(Residual.T, Residual) < 1:
                        s = z2.copy()
                        Residual = Residual_z2.copy()
                        Ps = Pz2.copy()

                if acceleration > 0:
                    #[s Residual] = MySubsetCG(x,s,P,Pt,find(s~=0),1e-9,0,CGSteps);
                    #Ps           = P(s);
                    raise RuntimeError('Large-scale not implemented yet')


                # Calculate step-size requirement
                omega = (np.linalg.norm(s - s_old, 2) / np.linalg.norm(Ps - old_Ps, 2)) ** 2


        else:  #Mu ~=0;
            # Use fixed step size

            if Count > 1 and acceleration == 0:
                s_very_old = s_old.copy()

            s_old = s.copy()
            s = s + MU * Pt(Residual)
            #[ssort sortind]     =   sort(abs(s),'descend');
            #s(sortind(M+1:end)) =   0;
            sortind = np.argsort(np.abs(s))[::-1]
            ssort = np.abs(s)[sortind]
            s[sortind[M:]] = 0
            if Count > 1 and acceleration == 0:
                very_old_Ps = old_Ps.copy()

            old_Ps = Ps.copy()

            Ps = P(s)
            Residual = measurements - Ps

            if Count > 2 and acceleration == 0:
                # 1st over-relaxation
                Dif = (Ps - old_Ps)
                a1 = np.dot(Dif.T, Residual) / np.dot(Dif.T, Dif)
                z1 = s + a1 * (s - s_old)
                Pz1 = (1 + a1) * Ps - a1 * old_Ps
                Residual_z1 = measurements - Pz1

                # 2nd over-relaxation
                Dif = (Pz1 - very_old_Ps)
                a2 = np.dot(Dif.T, Residual_z1) / np.dot(Dif.T, Dif)
                z2 = z1 + a2 * (z1 - s_very_old)

                # Threshold z2
                #[z2sort sortind]     =   sort(abs(z2),'descend');
                #z2(sortind(M+1:end)) =   0;
                sortind = np.argsort(np.abs(z2))[::-1]
                z2sort = np.abs(z2)[sortind]
                z2[sortind[M:]] = 0

                Pz2 = P(z2)
                Residual_z2 = measurements - Pz2

                # Decide if z2 is any good


                if np.dot(Residual_z2.T, Residual_z2) / np.dot(Residual.T, Residual) < 1:
                    s = z2.copy()
                    Residual = Residual_z2.copy()
                    Ps = Pz2.copy()

            if acceleration > 0:
                #[s Residual] = MySubsetCG(x,s,P,Pt,find(s~=0),1e-9,0,CGSteps);
                #Ps           = P(s);
                raise RuntimeError('Large-scale not implemented yet')

        ERR = np.dot(Residual.T, Residual) / n

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #                        Are we done yet?
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # Convergence criterion modified by Kun Qiu
        gap = (np.linalg.norm(s - s_old, 2) ** 2) / m
        if gap < deltatol or iter >= maxiter:
            done = 1
        # Nic:
        relerror = np.linalg.norm(measurements - np.dot(dictionary, s), 2) / np.linalg.norm(measurements)
        if relerror < errortol:
            done = 1


        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #                    If not done, take another round
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if not done:
            iter = iter + 1
            oldERR = ERR

    return s
