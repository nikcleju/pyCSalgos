"""
l1min.py

Provides l1 minimization from l1magic toolbox
"""

# Author: Nicolae Cleju (translated to Python), Justin Romberg (original author of l1magic Matlab toolbox)
# License: BSD 3 clause

import math
import numpy as np
import scipy

from base import SparseSolver

class L1Min(SparseSolver):

    # All parameters related to the algorithm itself are given here.
    # The data and dictionary are given to the run() method
    def __init__(self, stopval, algorithm="l1magic"):

        # parameter check
        if stopval < 0:
            raise ValueError("stopping value is negative")

        self.stopval = stopval
        self.algorithm = algorithm

    def __str__(self):
        return "L1 Minimization ("+str(self.stopval)+", "+str(self.algorithm)+")"


    def solve(self, data, dictionary):
        return _l1min(data, dictionary, self.stopval, self.algorithm)


def _l1min(data, dictionary, stopval, algorithm):

    # Force data 2D
    if len(data.shape) == 1:
        data = np.atleast_2d(data)
        if data.shape[0] < data.shape[1]:
            data = np.transpose(data)

    N = dictionary.shape[1]
    Ndata = data.shape[1]
    coef = np.zeros((N, Ndata))

    if algorithm == "l1magic":
        for i in range(Ndata):
            if stopval == 0:
                coef[:,i] = l1eq_pd(np.zeros(N), dictionary, dictionary.T, data[:,i])
            elif stopval > 0:
                coef[:,i] = l1qc_logbarrier(np.zeros(N), dictionary, dictionary.T, data[:,i], stopval, lbtol=0.1*stopval)
            else:
                raise ValueError("stopping value is negative")
    else:
        raise ValueError("Algorithm '%s' does not exist", algorithm)
    return np.squeeze(coef)


class l1NotImplementedError(Exception):
    pass

# equality constraints:

def l1eq_pd(x0, A, At, b, pdtol=1e-3, pdmaxiter=50, cgtol=1e-8, cgmaxiter=200, verbose=False):

    """
    l1 minimization with l1magic toolbox
    :param x0:
    :param A:
    :param At:
    :param b:
    :param pdtol:
    :param pdmaxiter:
    :param cgtol:
    :param cgmaxiter:
    :param verbose:
    :return:
    """

    if np.linalg.norm(b) < 1e-16:
        return np.zeros_like(x0)

    if hasattr(A, '__call__'):
        largescale = True
    else:
        largescale = False

    N = x0.size

    alpha = 0.01
    beta = 0.5
    mu = 10

    gradf0 = np.hstack((np.zeros(N), np.ones(N)))

    # starting point --- make sure that it is feasible
    if largescale:
        raise l1NotImplementedError('Largescale not implemented yet!')
    else:
        if np.linalg.norm(np.dot(A,x0)-b) / np.linalg.norm(b) > cgtol:
            if verbose:
                print 'Starting point infeasible; using x0 = At*inv(AAt)*y.'
            try:
                w = scipy.linalg.solve(np.dot(A,A.T), b, sym_pos=True)
                hcond = 1.0/np.linalg.cond(np.dot(A,A.T))
            except scipy.linalg.LinAlgError:
                if verbose:
                    print 'A*At is ill-conditioned: cannot find starting point'
                xp = x0.copy()
                return xp
            if hcond < 1e-14:
                if verbose:
                    print 'A*At is ill-conditioned: cannot find starting point'
                xp = x0.copy()
                return xp
            x0 = np.dot(A.T, w)
    x = x0.copy()
    u = (0.95)*np.abs(x0) + (0.10)*np.abs(x0).max()

    # set up for the first iteration
    fu1 = x - u
    fu2 = -x - u
    lamu1 = -1/fu1
    lamu2 = -1/fu2
    if (largescale):
        raise l1NotImplementedError('Largescale not implemented yet!')
    else:
        v = np.dot(-A, lamu1-lamu2)
        Atv = np.dot(A.T, v)
        rpri = np.dot(A,x) - b

    sdg = -(np.dot(fu1,lamu1) + np.dot(fu2,lamu2))
    tau = mu*2*N/sdg

    rcent = np.hstack((-np.dot(lamu1,fu1),  -np.dot(lamu2,fu2))) - (1/tau)
    rdual = gradf0 + np.hstack((lamu1-lamu2, -lamu1-lamu2)) + np.hstack((Atv, np.zeros(N)))
    resnorm = np.linalg.norm(np.hstack((rdual, rcent, rpri)))

    pditer = 0
    done = (sdg < pdtol) or (pditer >= pdmaxiter)
    while not done:

        pditer = pditer + 1

        w1 = -1/tau*(-1/fu1 + 1/fu2) - Atv
        w2 = -1 - 1/tau*(1/fu1 + 1/fu2)
        w3 = -rpri

        sig1 = -lamu1/fu1 - lamu2/fu2
        sig2 = lamu1/fu1 - lamu2/fu2
        sigx = sig1 - sig2**2/sig1

        if largescale:
            raise l1NotImplementedError('Largescale not implemented yet!')
        else:
            w1p = -(w3 - np.dot(A,(w1/sigx - w2*sig2/(sigx*sig1))))
            H11p = np.dot(A, np.dot(np.diag(1/sigx),A.T))
            try:
                dv = scipy.linalg.solve(H11p, w1p, sym_pos=True)
                hcond = 1.0/np.linalg.cond(H11p)
            except scipy.linalg.LinAlgError:
                if verbose:
                    print 'Matrix ill-conditioned.  Returning previous iterate.  (See Section 4 of notes for more information.)'
                xp = x.copy()
                return xp
            if hcond < 1e-14:
                if verbose:
                    print 'Matrix ill-conditioned.  Returning previous iterate.  (See Section 4 of notes for more information.)'
                xp = x.copy()
                return xp

            dx = (w1 - w2*sig2/sig1 - np.dot(A.T,dv))/sigx
            Adx = np.dot(A,dx)
            Atdv = np.dot(A.T,dv)

        du = (w2 - sig2*dx)/sig1

        dlamu1 = (lamu1/fu1)*(-dx+du) - lamu1 - (1/tau)*1/fu1
        dlamu2 = (lamu2/fu2)*(dx+du) - lamu2 - 1/tau*1/fu2

        indp = np.nonzero(dlamu1 < 0)
        indn = np.nonzero(dlamu2 < 0)
        s = np.min(np.hstack((np.array([1]), -lamu1[indp]/dlamu1[indp], -lamu2[indn]/dlamu2[indn])))
        indp = np.nonzero((dx-du) > 0)
        indn = np.nonzero((-dx-du) > 0)
        s = (0.99)*np.min(np.hstack((np.array([s]), -fu1[indp]/(dx[indp]-du[indp]), -fu2[indn]/(-dx[indn]-du[indn]))))

        # backtracking line search
        suffdec = 0
        backiter = 0
        while not suffdec:
            xp = x + s*dx
            up = u + s*du
            vp = v + s*dv
            Atvp = Atv + s*Atdv
            lamu1p = lamu1 + s*dlamu1
            lamu2p = lamu2 + s*dlamu2
            fu1p = xp - up
            fu2p = -xp - up
            rdp = gradf0 + np.hstack((lamu1p-lamu2p, -lamu1p-lamu2p)) + np.hstack((Atvp, np.zeros(N)))
            rcp = np.hstack((-lamu1p*fu1p, -lamu2p*fu2p)) - (1/tau)
            rpp = rpri + s*Adx
            suffdec = (np.linalg.norm(np.hstack((rdp, rcp, rpp))) <= (1-alpha*s)*resnorm)
            s = beta*s
            backiter = backiter + 1
            if (backiter > 32):
                if verbose:
                    print 'Stuck backtracking, returning last iterate.  (See Section 4 of notes for more information.)'
                xp = x.copy()
                return xp

        # next iteration
        x = xp.copy()
        u = up.copy()
        v = vp.copy()
        Atv = Atvp.copy()
        lamu1 = lamu1p.copy()
        lamu2 = lamu2p.copy()
        fu1 = fu1p.copy()
        fu2 = fu2p.copy()

        # surrogate duality gap
        sdg = -(np.dot(fu1,lamu1) + np.dot(fu2,lamu2))
        tau = mu*2*N/sdg
        rpri = rpp.copy()
        rcent = np.hstack((-lamu1*fu1, -lamu2*fu2)) - (1/tau)
        rdual = gradf0 + np.hstack((lamu1-lamu2, -lamu1-lamu2)) + np.hstack((Atv, np.zeros(N)))
        resnorm = np.linalg.norm(np.hstack((rdual, rcent, rpri)))

        done = (sdg < pdtol) or (pditer >= pdmaxiter)

        if verbose:
            print 'Iteration =',pditer,', tau =',tau,', Primal =',np.sum(u),', PDGap =',sdg,', Dual res =',np.linalg.norm(rdual),', Primal res =',np.linalg.norm(rpri)
        if largescale:
            raise l1NotImplementedError('Largescale not implemented yet!')
        else:
            if verbose:
                print '                  H11p condition number =',hcond
    return xp


# quadratic constraints:

def cgsolve(A, b, tol, maxiter, verbose=1):

    if hasattr(A, '__call__'):
        implicit = True
    else:
        implicit = False

    x = np.zeros(b.size)
    r = b.copy()
    d = r.copy()
    delta = np.vdot(r,r)
    delta0 = np.vdot(b,b)
    numiter = 0
    bestx = x.copy()
    bestres = math.sqrt(delta/delta0)
    while (numiter < maxiter) and (delta > tol**2*delta0):

        if implicit:
            q = A(d)
        else:
            q = np.dot(A,d)

        alpha = delta/np.vdot(d,q)
        x = x + alpha*d

        if divmod(numiter+1,50)[1] == 0:
            if implicit:
                r = b - A(x)
            else:
                r = b - np.dot(A,x)
        else:
            r = r - alpha*q

        deltaold = delta;
        delta = np.vdot(r,r)
        beta = delta/deltaold;
        d = r + beta*d
        numiter = numiter + 1
        if (math.sqrt(delta/delta0) < bestres):
            bestx = x.copy()
            bestres = math.sqrt(delta/delta0)

        if ((verbose) and (divmod(numiter,verbose)[1]==0)):
            print 'cg: Iter = ',numiter,', Best residual = ',bestres,', Current residual = ',math.sqrt(delta/delta0)

    if (verbose):
        print 'cg: Iterations = ',numiter,', best residual = ',bestres
    x = bestx.copy()
    res = bestres
    iter = numiter

    return x,res,iter

def l1qc_newton(x0, u0, A, At, b, epsilon, tau, newtontol, newtonmaxiter, cgtol, cgmaxiter, verbose=False):

    if hasattr(A, '__call__'):
        largescale = True
    else:
        largescale = False

    # line search parameters
    alpha = 0.01
    beta = 0.5

    if not largescale:
        AtA = np.dot(A.T,A)

    # initial point
    x = x0.copy()
    u = u0.copy()
    if largescale:
        r = A(x) - b
    else:
        r = np.dot(A,x) - b

    fu1 = x - u
    fu2 = -x - u
    fe = 1.0/2*(np.vdot(r,r) - epsilon**2)
    f = u.sum() - (1.0/tau)*(np.log(-fu1).sum() + np.log(-fu2).sum() + math.log(-fe))

    niter = 0
    done = 0
    while not done:

        if largescale:
            atr = At(r)
        else:
            atr = np.dot(A.T,r)

        ntgz = 1.0/fu1 - 1.0/fu2 + 1.0/fe*atr
        ntgu = -tau - 1.0/fu1 - 1.0/fu2
        gradf = -(1.0/tau)*np.concatenate((ntgz, ntgu),0)

        sig11 = 1.0/(fu1**2) + 1.0/(fu2**2)
        sig12 = -1.0/(fu1**2) + 1.0/(fu2**2)
        sigx = sig11 - (sig12**2)/sig11

        w1p = ntgz - sig12/sig11*ntgu
        if largescale:
            h11pfun = lambda z: sigx*z - (1.0/fe)*At(A(z)) + 1.0/(fe**2)*np.dot(np.dot(atr.T,z),atr)
            dx,cgres,cgiter = cgsolve(h11pfun, w1p, cgtol, cgmaxiter, 0)
            if (cgres > 1.0/2):
                if verbose:
                    print 'Cannot solve system.  Returning previous iterate.  (See Section 4 of notes for more information.)'
                xp = x.copy()
                up = u.copy()
                return xp,up,niter
            Adx = A(dx)
        else:
            # Attention: atr is column vector, so atr*atr' means outer(atr,atr)
            H11p = np.diag(sigx) - (1.0/fe)*AtA + (1.0/fe)**2*np.outer(atr,atr)
            try:
                dx = scipy.linalg.solve(H11p, w1p, sym_pos=True)
                hcond = 1.0/np.linalg.cond(H11p)
            except scipy.linalg.LinAlgError:
                if verbose:
                    print 'Matrix ill-conditioned.  Returning previous iterate.  (See Section 4 of notes for more information.)'
                xp = x.copy()
                up = u.copy()
                return xp,up,niter
            if hcond < 1e-14:
                if verbose:
                    print 'Matrix ill-conditioned.  Returning previous iterate.  (See Section 4 of notes for more information.)'
                xp = x.copy()
                up = u.copy()
                return xp,up,niter

            Adx = np.dot(A,dx)
        du = (1.0/sig11)*ntgu - (sig12/sig11)*dx;

        # minimum step size that stays in the interior
        ifu1 = np.nonzero((dx-du)>0)
        ifu2 = np.nonzero((-dx-du)>0)
        aqe = np.dot(Adx.T,Adx)
        bqe = 2*np.dot(r.T,Adx)
        cqe = np.vdot(r,r) - epsilon**2
        smax = min(1,np.concatenate( (-fu1[ifu1]/(dx[ifu1]-du[ifu1]) , -fu2[ifu2]/(-dx[ifu2]-du[ifu2]) , np.array([ (-bqe + math.sqrt(bqe**2-4*aqe*cqe))/(2*aqe) ]) ) , 0).min())

        s = 0.99 * smax

        # backtracking line search
        suffdec = 0
        backiter = 0
        while not suffdec:
            xp = x + s*dx
            up = u + s*du
            rp = r + s*Adx
            fu1p = xp - up
            fu2p = -xp - up
            fep = 0.5*(np.vdot(rp,rp) - epsilon**2)
            fp = up.sum() - (1.0/tau)*(np.log(-fu1p).sum() + np.log(-fu2p).sum() + math.log(-fep))
            flin = f + alpha*s*np.dot(gradf.T , np.concatenate((dx,du),0))
            if fp <= flin:
                suffdec = True
            else:
                suffdec = False

            s = beta*s
            backiter = backiter + 1
            if (backiter > 32):
                if verbose:
                    print 'Stuck on backtracking line search, returning previous iterate.  (See Section 4 of notes for more information.)'
                xp = x.copy()
                up = u.copy()
                return xp,up,niter

        # set up for next iteration
        x = xp.copy()
        u = up.copy()
        r = rp.copy()
        fu1 = fu1p.copy()
        fu2 = fu2p.copy()
        fe = fep
        f = fp

        lambda2 = -np.dot(gradf.T , np.concatenate((dx,du),0))
        stepsize = s * np.linalg.norm(np.concatenate((dx,du),0))
        niter = niter + 1
        if lambda2/2.0 < newtontol or niter >= newtonmaxiter:
            done = 1
        else:
            done = 0

        if verbose:
            print 'Newton iter = ',niter,', Functional = ',f,', Newton decrement = ',lambda2/2.0,', Stepsize = ',stepsize

        if verbose:
            if largescale:
                print '                CG Res = ',cgres,', CG Iter = ',cgiter
            else:
                print '                  H11p condition number = ',hcond
    return xp,up,niter

def l1qc_logbarrier(x0, A, At, b, epsilon, lbtol=1e-3, mu=10, cgtol=1e-8, cgmaxiter=200, verbose=False):

    # Check if epsilon > 0. If epsilon is 0, the algorithm fails. You should run the algo with equality constraint instead
    if epsilon == 0:
        raise ValueError('Epsilon should be > 0!')

    if hasattr(A, '__call__'):
        largescale = True
    else:
        largescale = False

    newtontol = lbtol
    newtonmaxiter = 50

    N = x0.size

    # starting point --- make sure that it is feasible
    if largescale:
        if np.linalg.norm(A(x0) - b) > epsilon:
            if verbose:
                print 'Starting point infeasible; using x0 = At*inv(AAt)*y.'
            AAt = lambda z: A(At(z))
            # TODO: implement cgsolve
            w,cgres,cgiter = cgsolve(AAt, b, cgtol, cgmaxiter, 0)
            if (cgres > 1.0/2):
                if verbose:
                    print 'A*At is ill-conditioned: cannot find starting point'
                xp = x0.copy()
                return xp
            x0 = At(w)
    else:
        if np.linalg.norm( np.dot(A,x0) - b ) > epsilon:
            if verbose:
                print 'Starting point infeasible; using x0 = At*inv(AAt)*y.'
            try:
                w = scipy.linalg.solve(np.dot(A,A.T), b, sym_pos=True)
                hcond = 1.0/np.linalg.cond(np.dot(A,A.T))
            except scipy.linalg.LinAlgError:
                if verbose:
                    print 'A*At is ill-conditioned: cannot find starting point'
                xp = x0.copy()
                return xp
            if hcond < 1e-14:
                if verbose:
                    print 'A*At is ill-conditioned: cannot find starting point'
                xp = x0.copy()
                return xp
            x0 = np.dot(A.T, w)
    x = x0.copy()
    u = (0.95)*np.abs(x0) + (0.10)*np.abs(x0).max()

    if verbose:
        print 'Original l1 norm = ',np.abs(x0).sum(),'original functional = ',u.sum()

    # choose initial value of tau so that the duality gap after the first
    # step will be about the origial norm
    tau = max(((2*N+1.0)/np.abs(x0).sum()), 1)

    lbiter = math.ceil((math.log(2*N+1)-math.log(lbtol)-math.log(tau))/math.log(mu))
    if verbose:
        print 'Number of log barrier iterations = ',lbiter

    totaliter = 0

    # Added by Nic, to fix some crashing
    if lbiter == 0:
        xp = np.zeros(x0.size)

    for ii in np.arange(lbiter):

        xp,up,ntiter = l1qc_newton(x, u, A, At, b, epsilon, tau, newtontol, newtonmaxiter, cgtol, cgmaxiter)
        totaliter = totaliter + ntiter

        if verbose:
            print 'Log barrier iter = ',ii,', l1 = ',np.abs(xp).sum(),', functional = ',up.sum(),', tau = ',tau,', total newton iter = ',totaliter
        x = xp.copy()
        u = up.copy()

        tau = mu*tau

    return xp
