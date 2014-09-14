"""
nesta.py

Provides L1 minimization algorithms for analysis-based compressed sensing
"""

import math

import numpy as np

from base import AnalysisSparseSolver


class AnalysisL1Min(AnalysisSparseSolver):
    """
    Analysis-based :math:`\ell_1` minimization
    """

    # All parameters related to the algorithm itself are given here.
    # The data and dictionary are given to the solve() method
    def __init__(self, stopval, algorithm="nesta"):

        # parameter check
        if stopval <= 0:
            raise ValueError("sigmamin is negative or zero")

        self.stopval = stopval
        self.algorithm = algorithm

    def __str__(self):
        return "AnalysisL1Min ("+str(self.stopval)+", "+str(self.algorithm)+")"


    def solve(self, measurements, acqumatrix, operator, realdict=None):
        #return analysis_l1min(measurements, dictionary, self.stopval, self.algorithm)

        # Ensure measurements 2D
        if len(measurements.shape) == 1:
            measurements = np.atleast_2d(measurements)
            if measurements.shape[0] < measurements.shape[1]:
                measurements = np.transpose(measurements)

        N = acqumatrix.shape[1]
        Ndata = measurements.shape[1]
        outdata = np.zeros((N, Ndata))

        if self.algorithm == "nesta":
            U,S,V = np.linalg.svd(acqumatrix, full_matrices = True)
            V = V.T         # Make like Matlab
            m,n = acqumatrix.shape   # Make like Matlab
            S = np.hstack((np.diag(S), np.zeros((m, n-m))))

            muf = 1e-6
            optsUSV = {'U':U, 'S':S, 'V':V}
            opts = {'U':operator, 'Ut':operator.T.copy(), 'USV':optsUSV, 'TolVar':1e-8, 'Verbose':0}
            for i in range(Ndata):
                outdata[:, i] = nesta(acqumatrix, None, measurements[:,i], muf, self.stopval, opts)[0]
        else:
            raise ValueError("Algorithm '%s' does not exist", self.algorithm)
        return outdata



class NestaError(Exception):
    pass


def nesta(A,At,b,muf,delta,opts=None):
    # [xk,niter,residuals,outputData] =NESTA(A,At,b,muf,delta,opts)
    #
    # Solves a L1 minimization problem under a quadratic constraint using the
    # Nesterov algorithm, with continuation:
    #
    #     min_x || U x ||_1 s.t. ||y - Ax||_2 <= delta
    #
    # Continuation is performed by sequentially applying Nesterov's algorithm
    # with a decreasing sequence of values of  mu0 >= mu >= muf
    #
    # The primal prox-function is also adapted by accounting for a first guess
    # xplug that also tends towards x_muf
    #
    # The observation matrix A is a projector
    #
    # Inputs:   A and At - measurement matrix and adjoint (either a matrix, in which
    #               case At is unused, or function handles).  m x n dimensions.
    #           b   - Observed data, a m x 1 array
    #           muf - The desired value of mu at the last continuation step.
    #               A smaller mu leads to higher accuracy.
    #           delta - l2 error bound.  This enforces how close the variable
    #               must fit the observations b, i.e. || y - Ax ||_2 <= delta
    #               If delta = 0, enforces y = Ax
    #               Common heuristic: delta = sqrt(m + 2*sqrt(2*m))*sigma;
    #               where sigma=std(noise).
    #           opts -
    #               This is a structure that contains additional options,
    #               some of which are optional.
    #               The fieldnames are case insensitive.  Below
    #               are the possible fieldnames:
    #
    #               opts.xplug - the first guess for the primal prox-function, and
    #                 also the initial point for xk.  By default, xplug = At(b)
    #               opts.U and opts.Ut - Analysis/Synthesis operators
    #                 (either matrices of function handles).
    #               opts.normU - if opts.U is provided, this should be norm(U)
    #                   otherwise it will have to be calculated (potentially
    #                   expensive)
    #               opts.MaxIntIter - number of continuation steps.
    #                 default is 5
    #               opts.maxiter - max number of iterations in an inner loop.
    #                 default is 10,000
    #               opts.TolVar - tolerance for the stopping criteria
    #               opts.stopTest - which stopping criteria to apply
    #                   opts.stopTest == 1 : stop when the relative
    #                       change in the objective function is less than
    #                       TolVar
    #                   opts.stopTest == 2 : stop with the l_infinity norm
    #                       of difference in the xk variable is less
    #                       than TolVar
    #               opts.TypeMin - if this is 'L1' (default), then
    #                   minimizes a smoothed version of the l_1 norm.
    #                   If this is 'tv', then minimizes a smoothed
    #                   version of the total-variation norm.
    #                   The string is case insensitive.
    #               opts.Verbose - if this is 0 or false, then very
    #                   little output is displayed.  If this is 1 or true,
    #                   then output every iteration is displayed.
    #                   If this is a number p greater than 1, then
    #                   output is displayed every pth iteration.
    #               opts.fid - if this is 1 (default), the display is
    #                   the usual Matlab screen.  If this is the file-id
    #                   of a file opened with fopen, then the display
    #                   will be redirected to this file.
    #               opts.errFcn - if this is a function handle,
    #                   then the program will evaluate opts.errFcn(xk)
    #                   at every iteration and display the result.
    #                   ex.  opts.errFcn = @(x) norm( x - x_true )
    #               opts.outFcn - if this is a function handle,
    #                   then then program will evaluate opts.outFcn(xk)
    #                   at every iteration and save the results in outputData.
    #                   If the result is a vector (as opposed to a scalar),
    #                   it should be a row vector and not a column vector.
    #                   ex. opts.outFcn = @(x) [norm( x - xtrue, 'inf' ),...
    #                                           norm( x - xtrue) / norm(xtrue)]
    #               opts.AAtinv - this is an experimental new option.  AAtinv
    #                   is the inverse of AA^*.  This allows the use of a
    #                   matrix A which is not a projection, but only
    #                   for the noiseless (i.e. delta = 0) case.
    #               opts.USV - another experimental option.  This supercedes
    #                   the AAtinv option, so it is recommended that you
    #                   do not define AAtinv.  This allows the use of a matrix
    #                   A which is not a projection, and works for the
    #                   noisy ( i.e. delta > 0 ) case.
    #                   opts.USV should contain three fields:
    #                   opts.USV.U  is the U from [U,S,V] = svd(A)
    #                   likewise, opts.USV.S and opts.USV.V are S and V
    #                   from svd(A).  S may be a matrix or a vector.
    #
    #  Outputs:
    #           xk  - estimate of the solution x
    #           niter - number of iterations
    #           residuals - first column is the residual at every step,
    #               second column is the value of f_mu at every step
    #           outputData - a matrix, where each row r is the output
    #               from opts.outFcn, if supplied.
    #           opts - the structure containing the options that were used
    #
    # Written by: Jerome Bobin, Caltech
    # Email: bobin@acm.caltech.edu
    # Created: February 2009
    # Modified (version 1.0): May 2009, Jerome Bobin and Stephen Becker, Caltech
    # Modified (version 1.1): Nov 2009, Stephen Becker, Caltech
    #
    # NESTA Version 1.1
    #   See also Core_Nesterov

    #---- Set defaults
    opts,Verbose,userSet = setOpts(opts,'Verbose',True);
    opts,MaxIntIter,userSet = setOpts(opts,'MaxIntIter',5,1);
    opts,TypeMin,userSet = setOpts(opts,'TypeMin','L1');
    opts,TolVar,userSet = setOpts(opts,'tolvar',1e-5);
    opts,U,U_userSet = setOpts(opts,'U', lambda x: x );
    if not hasattr(U, '__call__'):
        opts,Ut,userSet = setOpts(opts,'Ut',None)
    else:
        opts,Ut,userSet = setOpts(opts,'Ut', lambda x: x )
    opts,xplug,userSet = setOpts(opts,'xplug',None);
    opts,normU,userSet = setOpts(opts,'normU',None);  # so we can tell if it's been set

    residuals = np.zeros((0,2))
    outputData = np.zeros(0)
    opts,AAtinv,userSet = setOpts(opts,'AAtinv',None);
    opts,USV,userSet = setOpts(opts,'USV',None);
    if len(USV.keys()):

        Q = USV['U']  # we can't use "U" as the variable name
        # since "U" already refers to the analysis operator
        S = USV['S']
        if S.ndim is 1:
            s = S
        else:
            s = np.diag(S)

        V = USV['V'];

    # -- We can handle non-projections IF a (fast) routine for computing
    #    the psuedo-inverse is available.
    #    We can handle a nonzero delta, but we need the full SVD
    if (AAtinv is None) and (USV is None):
        # Check if A is a partial isometry, i.e. if AA' = I
        z = np.random.randn(b.shape)
        if hasattr(A, '__call__'):
            AAtz = A(At(z))
        else:
            AAtz = np.dot(A, np.dot(A.T,z))

        if np.linalg.norm(AAtz - z) / np.linalg.norm(z) > 1e-8:
            print 'Measurement matrix A must be a partial isometry: AA''=I'
            raise NestaError('Measurement matrix A must be a partial isometry: AA''=I')

    # -- Find a initial guess if not already provided.
    #   Use least-squares solution: x_ref = A'*inv(A*A')*b
    # If A is a projection, the least squares solution is trivial
    if xplug is None or np.linalg.norm(xplug) < 1e-12:
        if USV is not None and AAtinv is None:
            AAtinv = np.dot(Q, np.dot(np.diag(s ** -2), Q.T))
        if AAtinv is not None:
            if delta > 0 and USV is None:
                print 'delta must be zero for non-projections'
                raise NestaError('delta must be zero for non-projections')
            if hasattr(AAtinv,'__call__'):
                x_ref = AAtinv(b)
            else:
                x_ref = np.dot(AAtinv , b)
        else:
            x_ref = b

        if hasattr(A,'__call__'):
            x_ref=At(x_ref);
        else:
            x_ref = np.dot(A.T, x_ref)

        if xplug is None:
            xplug = x_ref;
            # x_ref itself is used to calculate mu_0
            #   in the case that xplug has very small norm
    else:
        x_ref = xplug;

    # use x_ref, not xplug, to find mu_0
    if hasattr(U,'__call__'):
        Ux_ref = U(x_ref);
    else:
        Ux_ref = np.dot(U,x_ref)
    if TypeMin.lower() == 'l1':
        mu0 = 0.9*max(abs(Ux_ref))
    elif TypeMin.lower() == 'tv':
        print 'Nic: TODO: not implemented yet'
        raise NestaError('Nic: TODO: not implemented yet')

    # -- If U was set by the user and normU not supplied, then calcuate norm(U)
    if U_userSet and normU is None:
        # simple case: U*U' = I or U'*U = I, in which case norm(U) = 1
        z = np.random.standard_normal(xplug.shape)
        if hasattr(U,'__call__'):
            UtUz = Ut(U(z))
        else:
            UtUz = np.dot(U.T, np.dot(U,z))

        if np.linalg.norm( UtUz - z )/np.linalg.norm(z) < 1e-8:
            normU = 1
        else:
            z = np.random.standard_normal(Ux_ref.shape)
            if hasattr(U,'__call__'):
                UUtz = U(Ut(z))
            else:
                UUtz = np.dot(U, np.dot(U.T,z))
            if np.linalg.norm( UUtz - z )/np.linalg.norm(z) < 1e-8:
                normU = 1;

        if normU is None:
            # have to actually calculate the norm
            if hasattr(U,'__call__'):
                normU,cnt = my_normest(U,Ut,xplug.size,1e-3,30)
                if cnt == 30:
                    print 'Warning: norm(U) may be inaccurate'
            else:
                mU,nU = U.shape
                if mU < nU:
                    UU = np.dot(U,U.T)
                else:
                    UU = np.dot(U.T,U)
                # last resort is to call MATLAB's "norm", which is slow
                if np.linalg.norm( UU - np.diag(np.diag(UU)),'fro') < 100*np.finfo(float).eps:
                    # this means the matrix is diagonal, so norm is easy:
                    normU = math.sqrt( max(abs(np.diag(UU))) )

                # Nic: TODO: sparse not implemented
                #elif issparse(UU)
                #    normU = sqrt( normest(UU) );
                else:
                    if min(U.shape) > 2000:
                        # norm(randn(2000)) takes about 5 seconds on my PC
                        print 'Warning: calculation of norm(U) may be slow'
                    normU = math.sqrt( np.linalg.norm(UU, 2) );
        opts['normU'] = normU

    niter = 0;
    Gamma = (muf/mu0)**(1.0/MaxIntIter);
    mu = mu0;
    Gammat = (TolVar/0.1)**(1.0/MaxIntIter);
    TolVar = 0.1;

    for n1 in np.arange(MaxIntIter):

        mu = mu*Gamma;
        TolVar=TolVar*Gammat;
        opts['TolVar'] = TolVar;
        opts['xplug'] = xplug;
        if Verbose:
            print '   Beginning', opts['TypeMin'],'Minimization; mu =',mu

        xk,niter_int,res,out,optsOut = Core_Nesterov(A,At,b,mu,delta,opts)

        xplug = xk.copy();
        niter = niter_int + niter;

        residuals = np.vstack((residuals,res))
        if out is not None:
            outputData = np.vstack((outputData, out))

    opts = optsOut.copy()

    return xk,niter,residuals,outputData,opts



#---- internal routine for setting defaults
def setOpts(opts,field,default,mn=None,mx=None):

    var = default
    # has the option already been set?
    if field in opts.keys():
        # see if there is a capitalization problem:
        for key in opts.keys():
            if key.lower() == field.lower():
                opts[field] = opts[key]
                # Don't delete because it is copied by reference!
                #del opts[key]
                break

    if field in opts.keys() and (opts[field] is not None):
        var = opts[field]
        userSet = True
    else:
        userSet = False
    # perform error checking, if desired
    if mn is not None:
        if var < mn:
            print 'Variable',field,'is',var,', should be at least',mn
            raise NestaError('setOpts error: value too small')
    if mx is not None:
        if var > mx:
            print 'Variable',field,'is',var,', should be at most',mx
            raise NestaError('setOpts error: value too large')
    opts[field] = var

    return opts,var,userSet

# Nic: TODO: implement TV
#---- internal routine for setting mu0 in the tv minimization case
#function th=ValMUTv(x)
#    #N = length(x);n = floor(sqrt(N));
#    N = b.size
#    n = floor(sqrt(N))
#    Dv = spdiags([reshape([-ones(n-1,n); zeros(1,n)],N,1) ...
#            reshape([zeros(1,n); ones(n-1,n)],N,1)], [0 1], N, N);
#        Dh = spdiags([reshape([-ones(n,n-1) zeros(n,1)],N,1) ...
#            reshape([zeros(n,1) ones(n,n-1)],N,1)], [0 n], N, N);
#        D = sparse([Dh;Dv]);
#
#
#    Dhx = Dh*x;
#    Dvx = Dv*x;
#
#    sk = sqrt(abs(Dhx).^2 + abs(Dvx).^2);
#    th = max(sk);
#
#end

#end #-- end of NESTA function

############ POWER METHOD TO ESTIMATE NORM ###############
# Copied from MATLAB's "normest" function, but allows function handles, not just sparse matrices
def my_normest(S,St,n,tol=1e-6, maxiter=20):
    #MY_NORMEST Estimate the matrix 2-norm via power method.
    if S is None:
        St = S  # we assume the matrix is symmetric;
    x = np.ones(n)
    cnt = 0
    e = np.linalg.norm(x)
    if e == 0:
        return e,cnt
    x = x/e
    e0 = 0
    while abs(e-e0) > tol*e and cnt < maxiter:
        e0 = e
        Sx = S(x)
        if (Sx!=0).sum() == 0:
            Sx = np.random.rand(Sx.size)
        e = np.linalg.norm(Sx)
        x = St(Sx)
        x = x/np.linalg.norm(x)
        cnt = cnt+1



def Core_Nesterov(A,At,b,mu,delta,opts):
    # [xk,niter,residuals,outputData,opts] =Core_Nesterov(A,At,b,mu,delta,opts)
    #
    # Solves a L1 minimization problem under a quadratic constraint using the
    # Nesterov algorithm, without continuation:
    #
    #     min_x || U x ||_1 s.t. ||y - Ax||_2 <= delta
    #
    # If continuation is desired, see the function NESTA.m
    #
    # The primal prox-function is also adapted by accounting for a first guess
    # xplug that also tends towards x_muf
    #
    # The observation matrix A is a projector
    #
    # Inputs:   A and At - measurement matrix and adjoint (either a matrix, in which
    #               case At is unused, or function handles).  m x n dimensions.
    #           b   - Observed data, a m x 1 array
    #           muf - The desired value of mu at the last continuation step.
    #               A smaller mu leads to higher accuracy.
    #           delta - l2 error bound.  This enforces how close the variable
    #               must fit the observations b, i.e. || y - Ax ||_2 <= delta
    #               If delta = 0, enforces y = Ax
    #               Common heuristic: delta = sqrt(m + 2*sqrt(2*m))*sigma;
    #               where sigma=std(noise).
    #           opts -
    #               This is a structure that contains additional options,
    #               some of which are optional.
    #               The fieldnames are case insensitive.  Below
    #               are the possible fieldnames:
    #
    #               opts.xplug - the first guess for the primal prox-function, and
    #                 also the initial point for xk.  By default, xplug = At(b)
    #               opts.U and opts.Ut - Analysis/Synthesis operators
    #                 (either matrices of function handles).
    #               opts.normU - if opts.U is provided, this should be norm(U)
    #               opts.maxiter - max number of iterations in an inner loop.
    #                 default is 10,000
    #               opts.TolVar - tolerance for the stopping criteria
    #               opts.stopTest - which stopping criteria to apply
    #                   opts.stopTest == 1 : stop when the relative
    #                       change in the objective function is less than
    #                       TolVar
    #                   opts.stopTest == 2 : stop with the l_infinity norm
    #                       of difference in the xk variable is less
    #                       than TolVar
    #               opts.TypeMin - if this is 'L1' (default), then
    #                   minimizes a smoothed version of the l_1 norm.
    #                   If this is 'tv', then minimizes a smoothed
    #                   version of the total-variation norm.
    #                   The string is case insensitive.
    #               opts.Verbose - if this is 0 or false, then very
    #                   little output is displayed.  If this is 1 or true,
    #                   then output every iteration is displayed.
    #                   If this is a number p greater than 1, then
    #                   output is displayed every pth iteration.
    #               opts.fid - if this is 1 (default), the display is
    #                   the usual Matlab screen.  If this is the file-id
    #                   of a file opened with fopen, then the display
    #                   will be redirected to this file.
    #               opts.errFcn - if this is a function handle,
    #                   then the program will evaluate opts.errFcn(xk)
    #                   at every iteration and display the result.
    #                   ex.  opts.errFcn = @(x) norm( x - x_true )
    #               opts.outFcn - if this is a function handle,
    #                   then then program will evaluate opts.outFcn(xk)
    #                   at every iteration and save the results in outputData.
    #                   If the result is a vector (as opposed to a scalar),
    #                   it should be a row vector and not a column vector.
    #                   ex. opts.outFcn = @(x) [norm( x - xtrue, 'inf' ),...
    #                                           norm( x - xtrue) / norm(xtrue)]
    #               opts.AAtinv - this is an experimental new option.  AAtinv
    #                   is the inverse of AA^*.  This allows the use of a
    #                   matrix A which is not a projection, but only
    #                   for the noiseless (i.e. delta = 0) case.
    #                   If the SVD of A is U*S*V', then AAtinv = U*(S^{-2})*U'.
    #               opts.USV - another experimental option.  This supercedes
    #                   the AAtinv option, so it is recommended that you
    #                   do not define AAtinv.  This allows the use of a matrix
    #                   A which is not a projection, and works for the
    #                   noisy ( i.e. delta > 0 ) case.
    #                   opts.USV should contain three fields:
    #                   opts.USV.U  is the U from [U,S,V] = svd(A)
    #                   likewise, opts.USV.S and opts.USV.V are S and V
    #                   from svd(A).  S may be a matrix or a vector.
    #  Outputs:
    #           xk  - estimate of the solution x
    #           niter - number of iterations
    #           residuals - first column is the residual at every step,
    #               second column is the value of f_mu at every step
    #           outputData - a matrix, where each row r is the output
    #               from opts.outFcn, if supplied.
    #           opts - the structure containing the options that were used
    #
    # Written by: Jerome Bobin, Caltech
    # Email: bobin@acm.caltech.edu
    # Created: February 2009
    # Modified: May 2009, Jerome Bobin and Stephen Becker, Caltech
    # Modified: Nov 2009, Stephen Becker
    #
    # NESTA Version 1.1
    #   See also NESTA

    #---- Set defaults
    # opts = [];

    opts,maxiter,userSet = setOpts(opts,'maxiter',10000,0);
    opts,TolVar,userSet = setOpts(opts,'TolVar',1e-5);
    opts,TypeMin,userSet = setOpts(opts,'TypeMin','L1');
    opts,Verbose,userSet = setOpts(opts,'Verbose',True);
    opts,errFcn,userSet = setOpts(opts,'errFcn',None);
    opts,outFcn,userSet = setOpts(opts,'outFcn',None);
    opts,stopTest,userSet = setOpts(opts,'stopTest',1,1,2);
    opts,U,userSet = setOpts(opts,'U',lambda x: x );
    if not hasattr(U,'__call__'):
        opts,Ut,userSet = setOpts(opts,'Ut',None);
    else:
        opts,Ut,userSet = setOpts(opts,'Ut', lambda x: x );
    opts,xplug,userSet = setOpts(opts,'xplug',None);
    opts,normU,userSet = setOpts(opts,'normU',1);

    if delta < 0:
        print 'delta must be greater or equal to zero'
        raise NestaError('delta must be greater or equal to zero')

    if hasattr(A,'__call__'):
        Atfun = At;
        Afun = A;
    else:
        Atfun = lambda x: np.dot(A.T,x)
        Afun = lambda x: np.dot(A,x)
    Atb = Atfun(b);

    opts,AAtinv,userSet = setOpts(opts,'AAtinv',None);
    opts,USV,userSet = setOpts(opts,'USV',None);
    if USV is not None:
        Q = USV['U'];  # we can't use "U" as the variable name
        # since "U" already refers to the analysis operator
        S = USV['S'];
        if S.ndim is 1:
            s = S
            S = np.diag(s)
        else:
            s = np.diag(S)

        V = USV['V'];
        if AAtinv is None:
            AAtinv = np.dot(Q, np.dot(np.diag(s ** -2), Q.T))

    # --- for A not a projection (experimental)
    if AAtinv is not None:
        if hasattr(AAtinv, '__call__'):
            AAtinv_fun = AAtinv;
        else:
            AAtinv_fun = lambda x: np.dot(AAtinv,x)

        AtAAtb = Atfun( AAtinv_fun(b) );

    else:
        AtAAtb = Atb;
        AAtinv_fun = lambda x: x;

    if xplug == None:
        xplug = AtAAtb.copy();

    #---- Initialization
    N = len(xplug)
    wk = np.zeros(N)
    xk = xplug.copy()


    #---- Init Variables
    Ak = 0.0;
    Lmu = normU/mu;
    yk = xk.copy();
    zk = xk.copy();
    fmean = np.finfo(float).tiny/10.0;
    OK = 0;
    n = math.floor(math.sqrt(N));

    #---- Computing Atb
    Atb = Atfun(b);
    Axk = Afun(xk);# only needed if you want to see the residuals


    #---- TV Minimization
    if TypeMin == 'TV':
        print 'Nic:TODO: TV minimization not yet implemented!'
        raise NestaError('Nic:TODO: TV minimization not yet implemented!')
    #if strcmpi(TypeMin,'TV')
    #    Lmu = 8*Lmu;
    #    Dv = spdiags([reshape([-ones(n-1,n); zeros(1,n)],N,1) ...
    #        reshape([zeros(1,n); ones(n-1,n)],N,1)], [0 1], N, N);
    #    Dh = spdiags([reshape([-ones(n,n-1) zeros(n,1)],N,1) ...
    #        reshape([zeros(n,1) ones(n,n-1)],N,1)], [0 n], N, N);
    #    D = sparse([Dh;Dv]);
    #end


    Lmu1 = 1.0/Lmu;
    lambdaY = 0.;
    lambdaZ = 0.;

    #---- setup data storage variables
    DISPLAY_ERROR = False
    RECORD_DATA = False
    outputData = None
    residuals = np.zeros((maxiter,2))
    if errFcn is not None:
        DISPLAY_ERROR = True
    if outFcn is not None:  # Output max number of arguments
        RECORD_DATA = True
        outputData = np.zeros(maxiter, outFcn(xplug).shape[1]);

    for k in np.arange(maxiter):

        #---- Dual problem

        if TypeMin == 'L1':
            df,fx,val,uk = Perform_L1_Constraint(xk,mu,U,Ut)

        # Nic: TODO: TV not implemented yet !
        #if strcmpi(TypeMin,'TV')  [df,fx] = Perform_TV_Constraint(xk,mu,Dv,Dh,D,U,Ut);end

        #---- Primal Problem

        #---- Updating yk

        #
        # yk = Argmin_x Lmu/2 ||x - xk||_l2^2 + <df,x-xk> s.t. ||b-Ax||_l2 < delta
        # Let xp be sqrt(Lmu) (x-xk), dfp be df/sqrt(Lmu), bp be sqrt(Lmu)(b- Axk) and deltap be sqrt(Lmu)delta
        # yk =  xk + 1/sqrt(Lmu) Argmin_xp 1/2 || xp ||_2^2 + <dfp,xp> s.t. || bp - Axp ||_2 < deltap
        #


        cp = xk - 1./Lmu*df;  # this is "q" in eq. (3.7) in the paper

        Acp = Afun( cp );
        if AAtinv is not None and USV is None:
            AtAcp = Atfun( AAtinv_fun( Acp ) );
        else:
            AtAcp = Atfun( Acp );

        residuals[k,0] = np.linalg.norm(b-Axk)
        residuals[k,1] = fx              # the value of the objective
        #--- if user has supplied a function, apply it to the iterate
        if RECORD_DATA:
            outputData[k,:] = outFcn(xk);

        if delta > 0:
            if USV is not None:
                # there are more efficient methods, but we're assuming
                # that A is negligible compared to U and Ut.
                # Here we make the change of variables x <-- x - xk
                #       and                            df <-- df/L
                dfp = -Lmu1*df;
                Adfp = -(Axk - Acp);
                bp = b - Axk;
                deltap = delta;
                # Check if we even need to project:
                if np.linalg.norm( Adfp - bp ) < deltap:
                    lambdaY = 0.
                    projIter = 0;
                    # i.e. projection = dfp;
                    yk = xk + dfp;
                    Ayk = Axk + Adfp;
                else:
                    lambdaY_old = lambdaY;
                    projection,projIter,lambdaY = fastProjection(Q,S,V,dfp,bp,deltap, .999*lambdaY_old )
                    if lambdaY > 0:
                        print 'lambda is positive!'
                        raise NestaError('lambda is positive!')
                    yk = xk + projection;
                    Ayk = Afun(yk);
                    # DEBUGGING
                    #                 if projIter == 50
                    #                     fprintf('\n Maxed out iterations at y\n');
                    #                     keyboard
                    #                 end
                    #end
            else:
                lambdaa = max(0,Lmu*(np.linalg.norm(b-Acp)/delta - 1))
                gamma = lambdaa/(lambdaa + Lmu);
                yk = lambdaa/Lmu*(1-gamma)*Atb + cp - gamma*AtAcp;
                # for calculating the residual, we'll avoid calling A()
                # by storing A(yk) here (using A'*A = I):
                Ayk = lambdaa/Lmu*(1-gamma)*b + Acp - gamma*Acp;
                #end
        else:
            # if delta is 0, the projection is simplified:
            yk = AtAAtb + cp - AtAcp;
            Ayk = b.copy();
            #end

            # DEBUGGING
            #     if norm( Ayk - b ) > (1.05)*delta
            #         fprintf('\nAyk failed projection test\n');
            #         keyboard;
            #     end

        #--- Stopping criterion

        if fmean.size == 1:
            qp = np.inf
        else:
            qp = abs(fx - np.mean(fmean))/np.mean(fmean);

        if stopTest == 1:
            # look at the relative change in function value
            if qp <= TolVar and OK:
                break
            if qp <= TolVar and not OK:
                OK = 1
        elif stopTest == 2:
            # look at the l_inf change from previous iterate
            if k >= 1 and np.linalg.norm( xk - xold, 'inf' ) <= TolVar:
                break

        fmean = np.hstack((fx,fmean));
        if (len(fmean) > 10):
            fmean = fmean[:10]



        #--- Updating zk

        apk = 0.5*(k+1);
        Ak = Ak + apk;
        tauk = 2.0/(k+3);

        wk =  apk*df + wk;

        #
        # zk = Argmin_x Lmu/2 ||b - Ax||_l2^2 + Lmu/2||x - xplug ||_2^2+ <wk,x-xk>
        #   s.t. ||b-Ax||_l2 < delta
        #

        cp = xplug - 1.0/Lmu*wk;

        Acp = Afun( cp );
        if AAtinv is not None and USV is None:
            AtAcp = Atfun( AAtinv_fun( Acp ) );
        else:
            AtAcp = Atfun( Acp );

        if delta > 0:
            if USV is not None:
                # Make the substitution wk <-- wk/K

                #             dfp = (xplug - Lmu1*wk);  # = cp
                #             Adfp= (Axplug - Acp);
                dfp = cp.copy()
                Adfp = Acp.copy()
                bp = b.copy();
                deltap = delta;
                #             dfp = SLmu*xplug - SLmu1*wk;
                #             bp = SLmu*b;
                #             deltap = SLmu*delta;

                # See if we even need to project:
                if np.linalg.norm( Adfp - bp ) < deltap:
                    zk = dfp.copy();
                    Azk = Adfp.copy();
                else:
                    projection,projIter,lambdaZ = fastProjection(Q,S,V,dfp,bp,deltap, .999*lambdaZ )
                    if lambdaZ > 0:
                        print 'lambda is positive!'
                        raise NestaError('lambda is positive!')
                    zk = projection.copy();
                    #             zk = SLmu1*projection;
                    Azk = Afun(zk);

                    # DEBUGGING:
                    #                 if projIter == 50
                    #                     fprintf('\n Maxed out iterations at z\n');
                    #                     keyboard
                    #                 end
                    #end
            else:
                lambdaa = max(0,Lmu*(np.linalg.norm(b-Acp)/delta - 1));
                gamma = lambdaa/(lambdaa + Lmu);
                zk = lambdaa/Lmu*(1-gamma)*Atb + cp - gamma*AtAcp;
                # for calculating the residual, we'll avoid calling A()
                # by storing A(zk) here (using A'*A = I):
                Azk = lambdaa/Lmu*(1-gamma)*b + Acp - gamma*Acp;
                #end
        else:
            # if delta is 0, this is simplified:
            zk = AtAAtb + cp - AtAcp;
            Azk = b;

            # DEBUGGING
            #     if norm( Ayk - b ) > (1.05)*delta
            #         fprintf('\nAzk failed projection test\n');
            #         keyboard;
            #     end

        #--- Updating xk

        xkp = tauk*zk + (1-tauk)*yk;
        xold = xk.copy();
        xk = xkp.copy();
        Axk = tauk*Azk + (1-tauk)*Ayk;

        if not np.mod(k,10):
            Axk = Afun(xk)   # otherwise slowly lose precision
            # DEBUG
            #     if norm(Axk - Afun(xk) ) > 1e-6, disp('error with Axk'); keyboard; end

        #--- display progress if desired
        if Verbose and not np.mod(k+1,Verbose):
            print 'Iter: ',k+1,'  ~ fmu: ',fx,' ~ Rel. Variation of fmu: ',qp,' ~ Residual:',residuals[k,0]
            #--- if user has supplied a function to calculate the error,
            # apply it to the current iterate and dislay the output:
            if DISPLAY_ERROR:
                print ' ~ Error:',errFcn(xk)

        if abs(fx)>1e20 or abs(residuals[k,0]) >1e20 or np.isnan(fx):
            print 'Nesta: possible divergence or NaN.  Bad estimate of ||A''A||?'
            raise NestaError('Nesta: possible divergence or NaN.  Bad estimate of ||A''A||?')


    niter = k+1;

    #-- truncate output vectors
    residuals = residuals[:niter,:]
    if RECORD_DATA:
        outputData = outputData[:niter,:]

    return xk,niter,residuals,outputData,opts


############ PERFORM THE L1 CONSTRAINT ##################

#function [df,fx,val,uk] = Perform_L1_Constraint(xk,mu,U,Ut)
def Perform_L1_Constraint(xk,mu,U,Ut):

    if hasattr(U,'__call__'):
        uk = U(xk);
    else:
        uk = np.dot(U,xk)
    fx = uk.copy()

    uk = uk / np.maximum(mu,abs(uk))
    val = np.real(np.vdot(uk,fx))
    fx = np.real(np.vdot(uk,fx) - mu/2.*np.linalg.norm(uk)**2);

    if hasattr(U,'__call__'):
        df = Ut(uk);
    else:
        df = np.dot(U.T,uk)
    return df,fx,val,uk

# Nic: TODO: TV not implemented yet!
############ PERFORM THE TV CONSTRAINT ##################
#function [df,fx] = Perform_TV_Constraint(xk,mu,Dv,Dh,D,U,Ut)
#    if isa(U,'function_handle')
#        x = U(xk);
#    else
#        x = U*xk;
#    end
#    df = zeros(size(x));
#
#    Dhx = Dh*x;
#    Dvx = Dv*x;
#
#    tvx = sum(sqrt(abs(Dhx).^2+abs(Dvx).^2));
#    w = max(mu,sqrt(abs(Dhx).^2 + abs(Dvx).^2));
#    uh = Dhx ./ w;
#    uv = Dvx ./ w;
#    u = [uh;uv];
#    fx = real(u'*D*x - mu/2 * 1/numel(u)*sum(u'*u));
#    if isa(Ut,'function_handle')
#        df = Ut(D'*u);
#    else
#        df = U'*(D'*u);
#    end
#end


def fastProjection( U, S, V, y, b, epsilon, lambda0=0, DISP=False ):
    # [x,niter,lambda] = fastProjection(U, S, V, y, b, epsilon, [lambda0], [DISP] )
    #
    # minimizes || x - y ||
    #   such that || Ax - b || <= epsilon
    #
    # where USV' = A (i.e the SVD of A)
    #
    # The optional input "lambda0" is a guess for the Lagrange parameter
    #
    # Warning: for speed, does not calculate A(y) to see if x = y is feasible
    #
    # NESTA Version 1.1
    #   See also Core_Nesterov

    # Written by Stephen Becker, September 2009, srbecker@caltech.edu

    DEBUG = True;
    # -- Parameters for Newton's method --
    MAXIT = 70;
    TOL = 1e-8 * epsilon;
    # TOL = max( TOL, 10*eps );  # we can't do better than machine precision

    m = U.shape[0]
    n = V.shape[0]
    mn = min(m,n);

    if S.size > mn**2:
        S = np.diag(np.diag(S))            # S should be a small square matrix
    r = S.shape[0] # S is square
    if U.shape[1] > r:
        U = U[:,:r]
    if V.shape[1] > r:
        V = V[:,:r]

    s = np.diag(S);
    s2 = s**2;

    # What we want to do:
    #   b = b - A*y;
    #   bb = U'*b;

    # if A doesn't have full row rank, then b may not be in the range
    if U.shape[0] > U.shape[1]:
        bRange = np.dot(U, np.dot(U.T, b))
        bNull = b - bRange;
        epsilon = math.sqrt( epsilon**2 - np.linalg.norm(bNull)**2 );
    b = np.dot(U.T,b) - np.dot(S, np.dot(V.T,y))

    b2 = abs(b)**2;  # for complex data
    bs2 = b2*s2;
    epsilon2 = epsilon**2;

    # The following routine need to be fast
    # For efficiency (at cost of transparency), we are writing the calculations
    # in a way that minimize number of operations.  The functions "f"
    # and "fp" represent f and its derivative.

    # f = @(lambda) sum( b2 .*(1-lambda*s2).^(-2) ) - epsilon^2;
    # fp = @(lambda) 2*sum( bs2 .*(1-lambda*s2).^(-3) );

    l = lambda0;
    oldff = 0;
    one = np.ones(m);
    alpha = 1;      # take full Newton steps
    for k in np.arange(MAXIT):
        # make f(l) and fp(l) as efficient as possible:
        ls = one/(one-l*s2)
        ls2 = ls**2;
        ls3 = ls2*ls;
        ff = np.dot(b2.conj(), ls2)   # should be .', not ', even for complex data
        ff = ff - epsilon2;
        fpl = 2 * np.dot(bs2.conj(),ls3)  # should be .', not ', even for complex data
        #     ff = f(l);    # this is a little slower
        #     fpl = fp(l);  # this is a little slower
        d = -ff/fpl;
        if DISP:
            print k,', lambda is ',l,', f(lambda) is ',ff,', f''(lambda) is',fpl
        if abs(ff) < TOL:
            break
        l_old = l
        if k>2 and ( abs(ff) > 10*abs(oldff+100) ): #|| abs(d) > 1e13 )
            l = 0;
            alpha = 1.0/2.0;
            #         oldff = f(0);
            oldff = b2.sum(); oldff = oldff - epsilon2;
            if DISP:
                print 'restarting'
        else:
            if alpha < 1:
                alpha = (alpha+1.0)/2.0
            l = l + alpha*d;
            oldff = ff
            if l > 0:
                l = 0;  # shouldn't be positive
                oldff = b2.sum()
                oldff = oldff - epsilon2;
                #end
        if l_old == l and l == 0:
            if DISP:
                print 'Making no progress; x = y is probably feasible'
            break;

    # if k == MAXIT && DEBUG, disp('maxed out iterations'); end
    if l < 0:
        xhat = np.dot(-l, s*b/( 1. - np.dot(l,s2) ) )
        x = np.dot(V,xhat) + y
    else:
        # y is already feasible, so no need to project
        l = 0;
        x = y.copy();

    return x,k,l