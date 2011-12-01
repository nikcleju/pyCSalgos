# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 16:55:20 2011

@author: ncleju
"""

import numpy
import math

class NestaError(Exception):
  pass

#function [xk,niter,residuals,outputData,opts] =NESTA(A,At,b,muf,delta,opts)
def NESTA(A,At,b,muf,delta,opts=None):
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
  
  #---------------------
  # Original Matab code:
  #
  #if nargin < 6, opts = []; end
  #if isempty(opts) && isnumeric(opts), opts = struct; end
  #
  ##---- Set defaults
  #fid = setOpts('fid',1);
  #Verbose = setOpts('Verbose',true);
  #function printf(varargin), fprintf(fid,varargin{:}); end
  #MaxIntIter = setOpts('MaxIntIter',5,1);
  #TypeMin = setOpts('TypeMin','L1');
  #TolVar = setOpts('tolvar',1e-5);
  #[U,U_userSet] = setOpts('U', @(x) x );
  #if ~isa(U,'function_handle')
  #    Ut = setOpts('Ut',[]);
  #else
  #    Ut = setOpts('Ut', @(x) x );
  #end
  #xplug = setOpts('xplug',[]);
  #normU = setOpts('normU',[]);  # so we can tell if it's been set
  #
  #residuals = []; outputData = [];
  #AAtinv = setOpts('AAtinv',[]);
  #USV = setOpts('USV',[]);
  #if ~isempty(USV)
  #    if isstruct(USV)
  #        Q = USV.U;  # we can't use "U" as the variable name
  #                    # since "U" already refers to the analysis operator
  #        S = USV.S;
  #        if isvector(S), s = S; #S = diag(s);
  #        else s = diag(S); end
  #        #V = USV.V;
  #    else
  #        error('opts.USV must be a structure');
  #    end
  #end
  #
  ## -- We can handle non-projections IF a (fast) routine for computing
  ##    the psuedo-inverse is available.
  ##    We can handle a nonzero delta, but we need the full SVD
  #if isempty(AAtinv) && isempty(USV)
  #    # Check if A is a partial isometry, i.e. if AA' = I
  #    z = randn(size(b));
  #    if isa(A,'function_handle'), AAtz = A(At(z));
  #    else AAtz = A*(A'*z); end
  #    if norm( AAtz - z )/norm(z) > 1e-8
  #        error('Measurement matrix A must be a partial isometry: AA''=I');
  #    end
  #end
  #
  ## -- Find a initial guess if not already provided.
  ##   Use least-squares solution: x_ref = A'*inv(A*A')*b
  ## If A is a projection, the least squares solution is trivial
  #if isempty(xplug) || norm(xplug) < 1e-12
  #    if ~isempty(USV) && isempty(AAtinv)
  #        AAtinv = Q*diag( s.^(-2) )*Q';
  #    end
  #    if ~isempty(AAtinv)
  #        if delta > 0 && isempty(USV)
  #            error('delta must be zero for non-projections');
  #        end
  #        if isa(AAtinv,'function_handle')
  #            x_ref = AAtinv(b);
  #        else
  #            x_ref = AAtinv * b;
  #        end
  #    else
  #        x_ref = b;
  #    end
  #    
  #    if isa(A,'function_handle')
  #        x_ref=At(x_ref);
  #    else
  #        x_ref = A'*x_ref;
  #    end
  #
  #    if isempty(xplug)
  #        xplug = x_ref;
  #    end
  #    # x_ref itself is used to calculate mu_0
  #    #   in the case that xplug has very small norm
  #else
  #    x_ref = xplug;
  #end
  #
  ## use x_ref, not xplug, to find mu_0
  #if isa(U,'function_handle')
  #    Ux_ref = U(x_ref);
  #else
  #    Ux_ref = U*x_ref;
  #end
  #switch lower(TypeMin)
  #    case 'l1'
  #        mu0 = 0.9*max(abs(Ux_ref));
  #    case 'tv'
  #        mu0 = ValMUTv(Ux_ref);
  #end
  #
  ## -- If U was set by the user and normU not supplied, then calcuate norm(U)
  #if U_userSet && isempty(normU)
  #    # simple case: U*U' = I or U'*U = I, in which case norm(U) = 1
  #    z = randn(size(xplug));
  #    if isa(U,'function_handle'), UtUz = Ut(U(z)); else UtUz = U'*(U*z); end
  #    if norm( UtUz - z )/norm(z) < 1e-8
  #        normU = 1;
  #    else
  #        z = randn(size(Ux_ref));
  #        if isa(U,'function_handle')
  #            UUtz = U(Ut(z)); 
  #        else
  #            UUtz = U*(U'*z);
  #        end
  #        if norm( UUtz - z )/norm(z) < 1e-8
  #            normU = 1;
  #        end
  #    end
  #    
  #    if isempty(normU)
  #        # have to actually calculate the norm
  #        if isa(U,'function_handle')
  #            [normU,cnt] = my_normest(U,Ut,length(xplug),1e-3,30);
  #            if cnt == 30, printf('Warning: norm(U) may be inaccurate\n'); end
  #        else
  #            [mU,nU] = size(U);
  #            if mU < nU, UU = U*U'; else UU = U'*U; end 
  #            # last resort is to call MATLAB's "norm", which is slow
  #            if norm( UU - diag(diag(UU)),'fro') < 100*eps
  #                # this means the matrix is diagonal, so norm is easy:
  #                normU = sqrt( max(abs(diag(UU))) );
  #            elseif issparse(UU)
  #                normU = sqrt( normest(UU) );
  #            else
  #                if min(size(U)) > 2000
  #                    # norm(randn(2000)) takes about 5 seconds on my PC
  #                    printf('Warning: calculation of norm(U) may be slow\n');
  #                end
  #                normU = sqrt( norm(UU) );
  #            end
  #        end
  #    end
  #    opts.normU = normU;
  #end
  #        
  #
  #niter = 0;
  #Gamma = (muf/mu0)^(1/MaxIntIter);
  #mu = mu0;
  #Gammat= (TolVar/0.1)^(1/MaxIntIter);
  #TolVar = 0.1;
  # 
  #for nl=1:MaxIntIter
  #    
  #    mu = mu*Gamma;
  #    TolVar=TolVar*Gammat;    opts.TolVar = TolVar;
  #    opts.xplug = xplug;
  #    if Verbose, printf('\tBeginning #s Minimization; mu = #g\n',opts.TypeMin,mu); end
  #    [xk,niter_int,res,out,optsOut] = Core_Nesterov(...
  #        A,At,b,mu,delta,opts);
  #    
  #    xplug = xk;
  #    niter = niter_int + niter;
  #    
  #    residuals = [residuals; res];
  #    outputData = [outputData; out];
  #
  #end
  #opts = optsOut;
  
  # End of original Matab code:
  #---------------------
  
  
  #if isempty(opts) && isnumeric(opts), opts = struct; end
  
  #---- Set defaults
  #fid = setOpts('fid',1);
  opts,Verbose,userSet = setOpts(opts,'Verbose',True);
  #function printf(varargin), fprintf(fid,varargin{:}); end
  opts,MaxIntIter,userSet = setOpts(opts,'MaxIntIter',5,1);
  opts,TypeMin,userSet = setOpts(opts,'TypeMin','L1');
  opts,TolVar,userSet = setOpts(opts,'tolvar',1e-5);
  #[U,U_userSet] = setOpts('U', @(x) x );
  opts,U,U_userSet = setOpts(opts,'U', lambda x: x );
  #if ~isa(U,'function_handle')
  if not hasattr(U, '__call__'):
      opts,Ut,userSet = setOpts(opts,'Ut',None)
  else:
      opts,Ut,userSet = setOpts(opts,'Ut', lambda x: x )
  #end
  opts,xplug,userSet = setOpts(opts,'xplug',None);
  opts,normU,userSet = setOpts(opts,'normU',None);  # so we can tell if it's been set
  
  #residuals = []; outputData = [];
  residuals = numpy.zeros((0,2))
  outputData = numpy.zeros(0)
  opts,AAtinv,userSet = setOpts(opts,'AAtinv',None);
  opts,USV,userSet = setOpts(opts,'USV',None);
  #if ~isempty(USV)
  if len(USV.keys()):
      #if isstruct(USV)
      
      Q = USV['U']  # we can't use "U" as the variable name
                  # since "U" already refers to the analysis operator
      S = USV['S']
      if S.ndim is 1:
        s = S
      else:
        s = numpy.diag(S)
      
      V = USV['V'];
      #else
      #    error('opts.USV must be a structure');
      #end
  #end
  
  # -- We can handle non-projections IF a (fast) routine for computing
  #    the psuedo-inverse is available.
  #    We can handle a nonzero delta, but we need the full SVD
  #if isempty(AAtinv) && isempty(USV)
  if (AAtinv is None) and (USV is None):
      # Check if A is a partial isometry, i.e. if AA' = I
      #z = randn(size(b));
      z = numpy.random.randn(b.shape)
      #if isa(A,'function_handle'), AAtz = A(At(z));
      #else AAtz = A*(A'*z); end
      if hasattr(A, '__call__'):
        AAtz = A(At(z))
      else:
        #AAtz = A*(A'*z)
        AAtz = numpy.dot(A, numpy.dot(A.T,z))
      
      #if norm( AAtz - z )/norm(z) > 1e-8
      if numpy.linalg.norm(AAtz - z) / numpy.linalg.norm(z) > 1e-8:
          #error('Measurement matrix A must be a partial isometry: AA''=I');
          print 'Measurement matrix A must be a partial isometry: AA''=I'
          raise NestaError('Measurement matrix A must be a partial isometry: AA''=I')
      #end
  #end
  
  # -- Find a initial guess if not already provided.
  #   Use least-squares solution: x_ref = A'*inv(A*A')*b
  # If A is a projection, the least squares solution is trivial
  #if isempty(xplug) || norm(xplug) < 1e-12
  if xplug is None or numpy.linalg.norm(xplug) < 1e-12:
      #if ~isempty(USV) && isempty(AAtinv)
      if USV is not None and AAtinv is None:
          #AAtinv = Q*diag( s.^(-2) )*Q';
          AAtinv = numpy.dot(Q, numpy.dot(numpy.diag(s ** -2), Q.T))
      #end
      #if ~isempty(AAtinv)
      if AAtinv is not None:
          #if delta > 0 && isempty(USV)
          if delta > 0 and USV is None:
              #error('delta must be zero for non-projections');
              print 'delta must be zero for non-projections'
              raise NesteError('delta must be zero for non-projections')
          #end
          #if isa(AAtinv,'function_handle')
          if hasattr(AAtinv,'__call__'):
              x_ref = AAtinv(b)
          else:
              x_ref = numpy.dot(AAtinv , b)
          #end
      else:
          x_ref = b
      #end
      
      #if isa(A,'function_handle')
      if hasattr(A,'__call__'):
          x_ref=At(x_ref);
      else:
          #x_ref = A'*x_ref;
          x_ref = numpy.dot(A.T, x_ref)
      #end
      
      #if isempty(xplug)
      if xplug is None:
          xplug = x_ref;
      #end
      # x_ref itself is used to calculate mu_0
      #   in the case that xplug has very small norm
  else:
      x_ref = xplug;
  #end
  
  # use x_ref, not xplug, to find mu_0
  #if isa(U,'function_handle')
  if hasattr(U,'__call__'):
      Ux_ref = U(x_ref);
  else:
      Ux_ref = numpy.dot(U,x_ref)
  #end
  #switch lower(TypeMin)
  #    case 'l1'
  #        mu0 = 0.9*max(abs(Ux_ref));
  #    case 'tv'
  #        mu0 = ValMUTv(Ux_ref);
  #end
  if TypeMin.lower() == 'l1':
    mu0 = 0.9*max(abs(Ux_ref))
  elif TypeMin.lower() == 'tv':
    #mu0 = ValMUTv(Ux_ref);
    print 'Nic: TODO: not implemented yet'
    raise NestaError('Nic: TODO: not implemented yet')
  
  # -- If U was set by the user and normU not supplied, then calcuate norm(U)
  #if U_userSet && isempty(normU)
  if U_userSet and normU is None:
      # simple case: U*U' = I or U'*U = I, in which case norm(U) = 1
      #z = randn(size(xplug));
      z = numpy.random.standard_normal(xplug.shape)
      #if isa(U,'function_handle'), UtUz = Ut(U(z)); else UtUz = U'*(U*z); end
      if hasattr(U,'__call__'):
        UtUz = Ut(U(z))
      else:
        UtUz = numpy.dot(U.T, numpy.dot(U,z))
      
      if numpy.linalg.norm( UtUz - z )/numpy.linalg.norm(z) < 1e-8:
          normU = 1;
      else:
          z = numpy.random.standard_normal(Ux_ref.shape)
          #if isa(U,'function_handle'):
          if hasattr(U,'__call__'):
              UUtz = U(Ut(z)); 
          else:
              #UUtz = U*(U'*z);
              UUtz = numpy.dot(U, numpy.dot(U.T,z))
          #end
          if numpy.linalg.norm( UUtz - z )/numpy.linalg.norm(z) < 1e-8:
              normU = 1;
          #end
      #end
      
      #if isempty(normU)
      if normU is None:
          # have to actually calculate the norm
          #if isa(U,'function_handle')
          if hasattr(U,'__call__'):
              #[normU,cnt] = my_normest(U,Ut,length(xplug),1e-3,30);
              normU,cnt = my_normest(U,Ut,xplug.size,1e-3,30)
              #if cnt == 30, printf('Warning: norm(U) may be inaccurate\n'); end
              if cnt == 30:
                print 'Warning: norm(U) may be inaccurate'
          else:
              mU,nU = U.shape
              if mU < nU:
                UU = numpy.dot(U,U.T)
              else:
                UU = numpy.dot(U.T,U)
              # last resort is to call MATLAB's "norm", which is slow
              #if norm( UU - diag(diag(UU)),'fro') < 100*eps
              if numpy.linalg.norm( UU - numpy.diag(numpy.diag(UU)),'fro') < 100*numpy.finfo(float).eps:
                  # this means the matrix is diagonal, so norm is easy:
                  #normU = sqrt( max(abs(diag(UU))) );
                  normU = math.sqrt( max(abs(numpy.diag(UU))) )
                  
              # Nic: TODO: sparse not implemented 
              #elif issparse(UU)
              #    normU = sqrt( normest(UU) );
              else:
                  if min(U.shape) > 2000:
                      # norm(randn(2000)) takes about 5 seconds on my PC
                      #printf('Warning: calculation of norm(U) may be slow\n');
                      print 'Warning: calculation of norm(U) may be slow'
                  #end
                  normU = math.sqrt( numpy.linalg.norm(UU, 2) );
              #end
          #end
      #end
      #opts.normU = normU;
      opts['normU'] = normU
  #end
  
  niter = 0;
  Gamma = (muf/mu0)**(1.0/MaxIntIter);
  mu = mu0;
  Gammat = (TolVar/0.1)**(1.0/MaxIntIter);
  TolVar = 0.1;
   
  #for nl=1:MaxIntIter
  for n1 in numpy.arange(MaxIntIter):
      
      mu = mu*Gamma;
      TolVar=TolVar*Gammat;
      opts['TolVar'] = TolVar;
      opts['xplug'] = xplug;
      #if Verbose, printf('\tBeginning #s Minimization; mu = #g\n',opts.TypeMin,mu); end
      if Verbose:
        #printf('\tBeginning #s Minimization; mu = #g\n',opts.TypeMin,mu)
        print '   Beginning', opts['TypeMin'],'Minimization; mu =',mu
      
      #[xk,niter_int,res,out,optsOut] = Core_Nesterov(A,At,b,mu,delta,opts);
      xk,niter_int,res,out,optsOut = Core_Nesterov(A,At,b,mu,delta,opts)
      
      xplug = xk.copy();
      niter = niter_int + niter;
      
      #residuals = [residuals; res];
      residuals = numpy.vstack((residuals,res))
      #outputData = [outputData; out];
      if out is not None:
        outputData = numpy.vstack((outputData, out))
  
  #end
  opts = optsOut.copy()
  
  return xk,niter,residuals,outputData,opts



#---- internal routine for setting defaults
#function [var,userSet] = setOpts(field,default,mn,mx)
def setOpts(opts,field,default,mn=None,mx=None):
  
    var = default
    # has the option already been set?
    #if ~isfield(opts,field) 
    if field in opts.keys():
        # see if there is a capitalization problem:
        #names = fieldnames(opts);
        #for i = 1:length(names)
        for key in opts.keys():
            #if strcmpi(names{i},field)
            if key.lower() == field.lower():
                #opts.(field) = opts.(names{i});
                opts[field] = opts[key]
                #opts = rmfield(opts,names{i});
                # Don't delete because it is copied by reference!
                #del opts[key]
                break
            #end
        #end
    #end
    
    #if isfield(opts,field) && ~isempty(opts.(field))
    if field in opts.keys() and (opts[field] is not None):
        #var = opts.(field);  # override the default
        var = opts[field]
        userSet = True
    else:
        userSet = False
    #end
    # perform error checking, if desired
    #if nargin >= 3 && ~isempty(mn)
    if mn is not None:
        if var < mn:
            #printf('Variable #s is #f, should be at least #f\n',...
            #    field,var,mn); error('variable out-of-bounds');
            print 'Variable',field,'is',var,', should be at least',mn
            raise NestaError('setOpts error: value too small')
        #end
    #end
    #if nargin >= 4 && ~isempty(mx)
    if mx is not None: 
        if var > mx:
            #printf('Variable #s is #f, should be at least #f\n',...
            #    field,var,mn); error('variable out-of-bounds');
            print 'Variable',field,'is',var,', should be at most',mx
            raise NestaError('setOpts error: value too large')
        #end
    #end
    #opts.(field) = var;
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
#function [e,cnt] = my_normest(S,St,n,tol, maxiter)
def my_normest(S,St,n,tol=1e-6, maxiter=20):
    #MY_NORMEST Estimate the matrix 2-norm via power method.
    #if nargin < 4, tol = 1.e-6; end
    #if nargin < 5, maxiter = 20; end
    #if isempty(St)
    if S is None:
        St = S  # we assume the matrix is symmetric;
    #end
    x = numpy.ones(n);
    cnt = 0;
    e = numpy.linalg.norm(x);
    #if e == 0, return, end
    if e == 0:
      return e,cnt
    x = x/e;
    e0 = 0;
    while abs(e-e0) > tol*e and cnt < maxiter:
       e0 = e;
       Sx = S(x);
       #if nnz(Sx) == 0
       if (Sx!=0).sum() == 0:
          Sx = numpy.random.rand(Sx.size);
       #end
       e = numpy.linalg.norm(Sx);
       x = St(Sx);
       x = x/numpy.linalg.norm(x);
       cnt = cnt+1;
    #end
#end



#function [xk,niter,residuals,outputData,opts] = Core_Nesterov(A,At,b,mu,delta,opts)
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

  #---------------------
  # Original Matab code:

  #fid = setOpts('fid',1);
  #function printf(varargin), fprintf(fid,varargin{:}); end
  #maxiter = setOpts('maxiter',10000,0);
  #TolVar = setOpts('TolVar',1e-5);
  #TypeMin = setOpts('TypeMin','L1');
  #Verbose = setOpts('Verbose',true);
  #errFcn = setOpts('errFcn',[]);
  #outFcn = setOpts('outFcn',[]);
  #stopTest = setOpts('stopTest',1,1,2);
  #U = setOpts('U', @(x) x );
  #if ~isa(U,'function_handle')
  #    Ut = setOpts('Ut',[]);
  #else
  #    Ut = setOpts('Ut', @(x) x );
  #end
  #xplug = setOpts('xplug',[]);
  #normU = setOpts('normU',1);
  #
  #if delta < 0, error('delta must be greater or equal to zero'); end
  #
  #if isa(A,'function_handle')
  #    Atfun = At;
  #    Afun = A;
  #else
  #    Atfun = @(x) A'*x;
  #    Afun = @(x) A*x;
  #end
  #Atb = Atfun(b);
  #
  #AAtinv = setOpts('AAtinv',[]);
  #USV = setOpts('USV',[]);
  #if ~isempty(USV)
  #    if isstruct(USV)
  #        Q = USV.U;  # we can't use "U" as the variable name
  #                    # since "U" already refers to the analysis operator
  #        S = USV.S;
  #        if isvector(S), s = S; S = diag(s);
  #        else s = diag(S); end
  #        V = USV.V;
  #    else
  #        error('opts.USV must be a structure');
  #    end
  #    if isempty(AAtinv)
  #        AAtinv = Q*diag( s.^(-2) )*Q';
  #    end
  #end
  ## --- for A not a projection (experimental)
  #if ~isempty(AAtinv)
  #    if isa(AAtinv,'function_handle')
  #        AAtinv_fun = AAtinv;
  #    else
  #        AAtinv_fun = @(x) AAtinv * x;
  #    end
  #    
  #    AtAAtb = Atfun( AAtinv_fun(b) );
  #
  #else
  #    # We assume it's a projection
  #    AtAAtb = Atb;
  #    AAtinv_fun = @(x) x;
  #end
  #
  #if isempty(xplug)
  #    xplug = AtAAtb; 
  #end
  #
  ##---- Initialization
  #N = length(xplug);
  #wk = zeros(N,1); 
  #xk = xplug;
  #
  #
  ##---- Init Variables
  #Ak= 0;
  #Lmu = normU/mu;
  #yk = xk;
  #zk = xk;
  #fmean = realmin/10;
  #OK = 0;
  #n = floor(sqrt(N));
  #
  ##---- Computing Atb
  #Atb = Atfun(b);
  #Axk = Afun(xk);# only needed if you want to see the residuals
  ## Axplug = Axk;
  #
  #
  ##---- TV Minimization
  #if strcmpi(TypeMin,'TV')
  #    Lmu = 8*Lmu;
  #    Dv = spdiags([reshape([-ones(n-1,n); zeros(1,n)],N,1) ...
  #        reshape([zeros(1,n); ones(n-1,n)],N,1)], [0 1], N, N);
  #    Dh = spdiags([reshape([-ones(n,n-1) zeros(n,1)],N,1) ...
  #        reshape([zeros(n,1) ones(n,n-1)],N,1)], [0 n], N, N);
  #    D = sparse([Dh;Dv]);
  #end
  #
  #
  #Lmu1 = 1/Lmu;
  ## SLmu = sqrt(Lmu);
  ## SLmu1 = 1/sqrt(Lmu);
  #lambdaY = 0;
  #lambdaZ = 0;
  #
  ##---- setup data storage variables
  #[DISPLAY_ERROR, RECORD_DATA] = deal(false);
  #outputData = deal([]);
  #residuals = zeros(maxiter,2);
  #if ~isempty(errFcn), DISPLAY_ERROR = true; end
  #if ~isempty(outFcn) && nargout >= 4
  #    RECORD_DATA = true;
  #    outputData = zeros(maxiter, size(outFcn(xplug),2) );
  #end
  #
  #for k = 0:maxiter-1,
  #    
  #   #---- Dual problem
  #   
  #   if strcmpi(TypeMin,'L1')  [df,fx,val,uk] = Perform_L1_Constraint(xk,mu,U,Ut);end
  #   
  #   if strcmpi(TypeMin,'TV')  [df,fx] = Perform_TV_Constraint(xk,mu,Dv,Dh,D,U,Ut);end
  #   
  #   #---- Primal Problem
  #   
  #   #---- Updating yk 
  #    
  #    #
  #    # yk = Argmin_x Lmu/2 ||x - xk||_l2^2 + <df,x-xk> s.t. ||b-Ax||_l2 < delta
  #    # Let xp be sqrt(Lmu) (x-xk), dfp be df/sqrt(Lmu), bp be sqrt(Lmu)(b- Axk) and deltap be sqrt(Lmu)delta
  #    # yk =  xk + 1/sqrt(Lmu) Argmin_xp 1/2 || xp ||_2^2 + <dfp,xp> s.t. || bp - Axp ||_2 < deltap
  #    #
  #    
  #    
  #    cp = xk - 1/Lmu*df;  # this is "q" in eq. (3.7) in the paper
  #    
  #    Acp = Afun( cp );
  #    if ~isempty(AAtinv) && isempty(USV)
  #        AtAcp = Atfun( AAtinv_fun( Acp ) );
  #    else
  #        AtAcp = Atfun( Acp );
  #    end
  #    
  #    residuals(k+1,1) = norm( b-Axk);    # the residual
  #    residuals(k+1,2) = fx;              # the value of the objective
  #    #--- if user has supplied a function, apply it to the iterate
  #    if RECORD_DATA
  #        outputData(k+1,:) = outFcn(xk);
  #    end
  #    
  #    if delta > 0
  #        if ~isempty(USV)
  #            # there are more efficient methods, but we're assuming
  #            # that A is negligible compared to U and Ut.
  #            # Here we make the change of variables x <-- x - xk
  #            #       and                            df <-- df/L
  #            dfp = -Lmu1*df;  Adfp = -(Axk - Acp);
  #            bp = b - Axk;
  #            deltap = delta;
  #            # Check if we even need to project:
  #            if norm( Adfp - bp ) < deltap
  #                lambdaY = 0;  projIter = 0;
  #                # i.e. projection = dfp;
  #                yk = xk + dfp;
  #                Ayk = Axk + Adfp;
  #            else
  #                lambdaY_old = lambdaY;
  #                [projection,projIter,lambdaY] = fastProjection(Q,S,V,dfp,bp,...
  #                    deltap, .999*lambdaY_old );
  #                if lambdaY > 0, disp('lambda is positive!'); keyboard; end
  #                yk = xk + projection;
  #                Ayk = Afun(yk);
  #                # DEBUGGING
  ##                 if projIter == 50
  ##                     fprintf('\n Maxed out iterations at y\n');
  ##                     keyboard
  ##                 end
  #            end
  #        else
  #            lambda = max(0,Lmu*(norm(b-Acp)/delta - 1));gamma = lambda/(lambda + Lmu);
  #            yk = lambda/Lmu*(1-gamma)*Atb + cp - gamma*AtAcp;
  #            # for calculating the residual, we'll avoid calling A()
  #            # by storing A(yk) here (using A'*A = I):
  #            Ayk = lambda/Lmu*(1-gamma)*b + Acp - gamma*Acp;
  #        end
  #    else
  #        # if delta is 0, the projection is simplified:
  #        yk = AtAAtb + cp - AtAcp;
  #        Ayk = b;
  #    end
  #
  #    # DEBUGGING
  ##     if norm( Ayk - b ) > (1.05)*delta
  ##         fprintf('\nAyk failed projection test\n');
  ##         keyboard;
  ##     end
  #    
  #    #--- Stopping criterion
  #    qp = abs(fx - mean(fmean))/mean(fmean);
  #    
  #    switch stopTest
  #        case 1
  #            # look at the relative change in function value
  #            if qp <= TolVar && OK; break;end
  #            if qp <= TolVar && ~OK; OK=1; end
  #        case 2
  #            # look at the l_inf change from previous iterate
  #            if k >= 1 && norm( xk - xold, 'inf' ) <= TolVar
  #                break
  #            end
  #    end
  #    fmean = [fx,fmean];
  #    if (length(fmean) > 10) fmean = fmean(1:10);end
  #    
  #
  #    
  #    #--- Updating zk
  #  
  #    apk =0.5*(k+1);
  #    Ak = Ak + apk; 
  #    tauk = 2/(k+3); 
  #    
  #    wk =  apk*df + wk;
  #    
  #    #
  #    # zk = Argmin_x Lmu/2 ||b - Ax||_l2^2 + Lmu/2||x - xplug ||_2^2+ <wk,x-xk> 
  #    #   s.t. ||b-Ax||_l2 < delta
  #    #
  #    
  #    cp = xplug - 1/Lmu*wk;
  #    
  #    Acp = Afun( cp );
  #    if ~isempty( AAtinv ) && isempty(USV)
  #        AtAcp = Atfun( AAtinv_fun( Acp ) );
  #    else
  #        AtAcp = Atfun( Acp );
  #    end
  #    
  #    if delta > 0
  #        if ~isempty(USV)
  #            # Make the substitution wk <-- wk/K
  #                 
  ##             dfp = (xplug - Lmu1*wk);  # = cp
  ##             Adfp= (Axplug - Acp);
  #            dfp = cp; Adfp = Acp; 
  #            bp = b;
  #            deltap = delta;            
  ##             dfp = SLmu*xplug - SLmu1*wk;
  ##             bp = SLmu*b;
  ##             deltap = SLmu*delta;
  #
  #            # See if we even need to project:
  #            if norm( Adfp - bp ) < deltap
  #                zk = dfp;
  #                Azk = Adfp;
  #            else
  #                [projection,projIter,lambdaZ] = fastProjection(Q,S,V,dfp,bp,...
  #                    deltap, .999*lambdaZ );
  #                if lambdaZ > 0, disp('lambda is positive!'); keyboard; end
  #                zk = projection;
  #                #             zk = SLmu1*projection;
  #                Azk = Afun(zk);
  #            
  #                # DEBUGGING:
  ##                 if projIter == 50
  ##                     fprintf('\n Maxed out iterations at z\n');
  ##                     keyboard
  ##                 end
  #            end
  #        else
  #            lambda = max(0,Lmu*(norm(b-Acp)/delta - 1));gamma = lambda/(lambda + Lmu);
  #            zk = lambda/Lmu*(1-gamma)*Atb + cp - gamma*AtAcp;
  #            # for calculating the residual, we'll avoid calling A()
  #            # by storing A(zk) here (using A'*A = I):
  #            Azk = lambda/Lmu*(1-gamma)*b + Acp - gamma*Acp;
  #        end
  #    else
  #        # if delta is 0, this is simplified:
  #        zk = AtAAtb + cp - AtAcp;
  #        Azk = b;
  #    end
  #    
  #    # DEBUGGING
  ##     if norm( Ayk - b ) > (1.05)*delta
  ##         fprintf('\nAzk failed projection test\n');
  ##         keyboard;
  ##     end
  #
  #    #--- Updating xk
  #    
  #    xkp = tauk*zk + (1-tauk)*yk;
  #    xold = xk;
  #    xk=xkp; 
  #    Axk = tauk*Azk + (1-tauk)*Ayk;
  #    
  #    if ~mod(k,10), Axk = Afun(xk); end   # otherwise slowly lose precision
  #    # DEBUG
  ##     if norm(Axk - Afun(xk) ) > 1e-6, disp('error with Axk'); keyboard; end
  #    
  #    #--- display progress if desired
  #    if ~mod(k+1,Verbose )
  #        printf('Iter: #3d  ~ fmu: #.3e ~ Rel. Variation of fmu: #.2e ~ Residual: #.2e',...
  #            k+1,fx,qp,residuals(k+1,1) ); 
  #        #--- if user has supplied a function to calculate the error,
  #        # apply it to the current iterate and dislay the output:
  #        if DISPLAY_ERROR, printf(' ~ Error: #.2e',errFcn(xk)); end
  #        printf('\n');
  #    end
  #    if abs(fx)>1e20 || abs(residuals(k+1,1)) >1e20 || isnan(fx)
  #        error('Nesta: possible divergence or NaN.  Bad estimate of ||A''A||?');
  #    end
  #
  #end
  #
  #niter = k+1; 
  #
  ##-- truncate output vectors
  #residuals = residuals(1:niter,:);
  #if RECORD_DATA,     outputData = outputData(1:niter,:); end

  # End of original Matab code
  #---------------------

  #fid = setOpts('fid',1);
  #function printf(varargin), fprintf(fid,varargin{:}); end
  opts,maxiter,userSet = setOpts(opts,'maxiter',10000,0);
  opts,TolVar,userSet = setOpts(opts,'TolVar',1e-5);
  opts,TypeMin,userSet = setOpts(opts,'TypeMin','L1');
  opts,Verbose,userSet = setOpts(opts,'Verbose',True);
  opts,errFcn,userSet = setOpts(opts,'errFcn',None);
  opts,outFcn,userSet = setOpts(opts,'outFcn',None);
  opts,stopTest,userSet = setOpts(opts,'stopTest',1,1,2);
  opts,U,userSet = setOpts(opts,'U',lambda x: x );
  #if ~isa(U,'function_handle')
  if not hasattr(U,'__call__'):
      opts,Ut,userSet = setOpts(opts,'Ut',None);
  else:
      opts,Ut,userSet = setOpts(opts,'Ut', lambda x: x );
  #end
  opts,xplug,userSet = setOpts(opts,'xplug',None);
  opts,normU,userSet = setOpts(opts,'normU',1);
  
  if delta < 0:
    print 'delta must be greater or equal to zero'
    raise NestaError('delta must be greater or equal to zero')
  
  if hasattr(A,'__call__'):
      Atfun = At;
      Afun = A;
  else:
      Atfun = lambda x: numpy.dot(A.T,x)
      Afun = lambda x: numpy.dot(A,x)
  #end
  Atb = Atfun(b);
  
  opts,AAtinv,userSet = setOpts(opts,'AAtinv',None);
  opts,USV,userSet = setOpts(opts,'USV',None);
  if USV is not None:
      #if isstruct(USV)
      Q = USV['U'];  # we can't use "U" as the variable name
                  # since "U" already refers to the analysis operator
      S = USV['S'];
      #if isvector(S), s = S; S = diag(s);
      #else s = diag(S); end
      if S.ndim is 1:
        s = S
        S = numpy.diag(s)
      else:
        s = numpy.diag(S)
        
      V = USV['V'];
      #else
      #    error('opts.USV must be a structure');
      #end
      #if isempty(AAtinv)
      if AAtinv is None:
          #AAtinv = Q*diag( s.^(-2) )*Q';
          AAtinv = numpy.dot(Q, numpy.dot(numpy.diag(s ** -2), Q.T))
      #end
  #end
  # --- for A not a projection (experimental)
  #if ~isempty(AAtinv)
  if AAtinv is not None:
      #if isa(AAtinv,'function_handle')
      if hasattr(AAtinv, '__call__'):
          AAtinv_fun = AAtinv;
      else:
          AAtinv_fun = lambda x: numpy.dot(AAtinv,x)
      #end
      
      AtAAtb = Atfun( AAtinv_fun(b) );
  
  else:
      # We assume it's a projection
      AtAAtb = Atb;
      AAtinv_fun = lambda x: x;
  #end
  
  if xplug == None:
      xplug = AtAAtb.copy(); 
  #end
  
  #---- Initialization
  #N = length(xplug);
  N = len(xplug)
  #wk = zeros(N,1); 
  wk = numpy.zeros(N)
  xk = xplug.copy()
  
  
  #---- Init Variables
  Ak = 0.0;
  Lmu = normU/mu;
  yk = xk.copy();
  zk = xk.copy();
  fmean = numpy.finfo(float).tiny/10.0;
  OK = 0;
  n = math.floor(math.sqrt(N));
  
  #---- Computing Atb
  Atb = Atfun(b);
  Axk = Afun(xk);# only needed if you want to see the residuals
  # Axplug = Axk;
  
  
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
  # SLmu = sqrt(Lmu);
  # SLmu1 = 1/sqrt(Lmu);
  lambdaY = 0.;
  lambdaZ = 0.;
  
  #---- setup data storage variables
  #[DISPLAY_ERROR, RECORD_DATA] = deal(false);
  DISPLAY_ERROR = False
  RECORD_DATA = False
  #outputData = deal([]);
  outputData = None
  residuals = numpy.zeros((maxiter,2))
  #if ~isempty(errFcn), DISPLAY_ERROR = true; end
  if errFcn is not None:
    DISPLAY_ERROR = True
  #if ~isempty(outFcn) && nargout >= 4
  if outFcn is not None:  # Output max number of arguments
      RECORD_DATA = True
      outputData = numpy.zeros(maxiter, outFcn(xplug).shape[1]);
  #end
  
  #for k = 0:maxiter-1,
  for k in numpy.arange(maxiter):
      
      #---- Dual problem
     
      #if strcmpi(TypeMin,'L1')  [df,fx,val,uk] = Perform_L1_Constraint(xk,mu,U,Ut);end
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
      #if ~isempty(AAtinv) && isempty(USV)
      if AAtinv is not None and USV is None:
          AtAcp = Atfun( AAtinv_fun( Acp ) );
      else:
          AtAcp = Atfun( Acp );
      #end
      
      #residuals(k+1,1) = norm( b-Axk);    # the residual
      residuals[k,0] = numpy.linalg.norm(b-Axk)
      #residuals(k+1,2) = fx;              # the value of the objective
      residuals[k,1] = fx
      #--- if user has supplied a function, apply it to the iterate
      if RECORD_DATA:
          outputData[k,:] = outFcn(xk);
      #end
      
      if delta > 0:
          #if ~isempty(USV)
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
              if numpy.linalg.norm( Adfp - bp ) < deltap:
                  lambdaY = 0.
                  projIter = 0;
                  # i.e. projection = dfp;
                  yk = xk + dfp;
                  Ayk = Axk + Adfp;
              else:
                  lambdaY_old = lambdaY;
                  #[projection,projIter,lambdaY] = fastProjection(Q,S,V,dfp,bp,deltap, .999*lambdaY_old );
                  projection,projIter,lambdaY = fastProjection(Q,S,V,dfp,bp,deltap, .999*lambdaY_old )
                  #if lambdaY > 0, disp('lambda is positive!'); keyboard; end
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
              lambdaa = max(0,Lmu*(numpy.linalg.norm(b-Acp)/delta - 1))
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
        qp = numpy.inf
      else:
        qp = abs(fx - numpy.mean(fmean))/numpy.mean(fmean);
      
      #switch stopTest
      #    case 1
      if stopTest == 1:
              # look at the relative change in function value
              #if qp <= TolVar && OK; break;end
              if qp <= TolVar and OK:
                break
              #if qp <= TolVar && ~OK; OK=1; end
              if qp <= TolVar and not OK:
                OK = 1
      elif stopTest == 2:
              # look at the l_inf change from previous iterate
              if k >= 1 and numpy.linalg.norm( xk - xold, 'inf' ) <= TolVar:
                  break
              #end
      #end
      #fmean = [fx,fmean];
      fmean = numpy.hstack((fx,fmean));
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
      #if ~isempty( AAtinv ) && isempty(USV)
      if AAtinv is not None and USV is None:
          AtAcp = Atfun( AAtinv_fun( Acp ) );
      else:
          AtAcp = Atfun( Acp );
      #end
      
      if delta > 0:
          #if ~isempty(USV)
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
              if numpy.linalg.norm( Adfp - bp ) < deltap:
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
              lambdaa = max(0,Lmu*(numpy.linalg.norm(b-Acp)/delta - 1));
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
      #end
      
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
      
      #if ~mod(k,10), Axk = Afun(xk); end   # otherwise slowly lose precision
      if not numpy.mod(k,10):
        Axk = Afun(xk)
      # DEBUG
  #     if norm(Axk - Afun(xk) ) > 1e-6, disp('error with Axk'); keyboard; end
      
      #--- display progress if desired
      #if ~mod(k+1,Verbose )
      if Verbose and not numpy.mod(k+1,Verbose):
          #printf('Iter: #3d  ~ fmu: #.3e ~ Rel. Variation of fmu: #.2e ~ Residual: #.2e',k+1,fx,qp,residuals(k+1,1) ); 
          print 'Iter: ',k+1,'  ~ fmu: ',fx,' ~ Rel. Variation of fmu: ',qp,' ~ Residual:',residuals[k,0]
          #--- if user has supplied a function to calculate the error,
          # apply it to the current iterate and dislay the output:
          #if DISPLAY_ERROR, printf(' ~ Error: #.2e',errFcn(xk)); end
          if DISPLAY_ERROR:
            print ' ~ Error:',errFcn(xk)
      #end
      if abs(fx)>1e20 or abs(residuals[k,0]) >1e20 or numpy.isnan(fx):
          #error('Nesta: possible divergence or NaN.  Bad estimate of ||A''A||?');
          print 'Nesta: possible divergence or NaN.  Bad estimate of ||A''A||?'
          raise NestaError('Nesta: possible divergence or NaN.  Bad estimate of ||A''A||?')
      #end
  
  #end
  
  niter = k+1; 
  
  #-- truncate output vectors
  residuals = residuals[:niter,:]
  #if RECORD_DATA,     outputData = outputData(1:niter,:); end
  if RECORD_DATA:
    outputData = outputData[:niter,:]
    
  return xk,niter,residuals,outputData,opts


############ PERFORM THE L1 CONSTRAINT ##################

#function [df,fx,val,uk] = Perform_L1_Constraint(xk,mu,U,Ut)
def Perform_L1_Constraint(xk,mu,U,Ut):

    #if isa(U,'function_handle')
    if hasattr(U,'__call__'):
        uk = U(xk);
    else:
        uk = numpy.dot(U,xk)
    #end
    fx = uk.copy()

    #uk = uk./max(mu,abs(uk));
    uk = uk / numpy.maximum(mu,abs(uk))
    #val = real(uk'*fx);
    val = numpy.real(numpy.vdot(uk,fx))
    #fx = real(uk'*fx - mu/2*norm(uk)^2);
    fx = numpy.real(numpy.vdot(uk,fx) - mu/2.*numpy.linalg.norm(uk)**2);

    #if isa(Ut,'function_handle')
    if hasattr(U,'__call__'):
        df = Ut(uk);
    else:
        #df = U'*uk;
        df = numpy.dot(U.T,uk)
    #end
    return df,fx,val,uk
#end

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


#function [x,k,l] = fastProjection( U, S, V, y, b, epsilon, lambda0, DISP )
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
  
  #---------------------
  # Original Matab code:
  
  #DEBUG = true;
  #if nargin < 8
  #    DISP = false;
  #end
  ## -- Parameters for Newton's method --
  #MAXIT = 70;
  #TOL = 1e-8 * epsilon;
  ## TOL = max( TOL, 10*eps );  # we can't do better than machine precision
  #
  #m = size(U,1);
  #n = size(V,1);
  #mn = min([m,n]);
  #if numel(S) > mn^2, S = diag(diag(S)); end  # S should be a small square matrix
  #r = size(S);
  #if size(U,2) > r, U = U(:,1:r); end
  #if size(V,2) > r, V = V(:,1:r); end
  #
  #s = diag(S);
  #s2 = s.^2;
  #
  ## What we want to do:
  ##   b = b - A*y;
  ##   bb = U'*b;
  #
  ## if A doesn't have full row rank, then b may not be in the range
  #if size(U,1) > size(U,2)
  #    bRange = U*(U'*b);
  #    bNull = b - bRange;
  #    epsilon = sqrt( epsilon^2 - norm(bNull)^2 );
  #end
  #b = U'*b - S*(V'*y);  # parenthesis is very important!  This is expensive.
  #    
  ## b2 = b.^2;
  #b2 = abs(b).^2;  # for complex data
  #bs2 = b2.*s2;
  #epsilon2 = epsilon^2;
  #
  ## The following routine need to be fast
  ## For efficiency (at cost of transparency), we are writing the calculations
  ## in a way that minimize number of operations.  The functions "f"
  ## and "fp" represent f and its derivative.
  #
  ## f = @(lambda) sum( b2 .*(1-lambda*s2).^(-2) ) - epsilon^2;
  ## fp = @(lambda) 2*sum( bs2 .*(1-lambda*s2).^(-3) );
  #if nargin < 7, lambda0 = 0; end
  #l = lambda0; oldff = 0;
  #one = ones(m,1);
  #alpha = 1;      # take full Newton steps
  #for k = 1:MAXIT
  #    # make f(l) and fp(l) as efficient as possible:
  #    ls = one./(one-l*s2);
  #    ls2 = ls.^2;
  #    ls3 = ls2.*ls;
  #    ff = b2.'*ls2; # should be .', not ', even for complex data
  #    ff = ff - epsilon2;
  #    fpl = 2*( bs2.'*ls3 );  # should be .', not ', even for complex data
  ##     ff = f(l);    # this is a little slower
  ##     fpl = fp(l);  # this is a little slower
  #    d = -ff/fpl;
  #    if DISP, fprintf('#2d, lambda is #5.2f, f(lambda) is #.2e, f''(lambda) is #.2e\n',...
  #            k,l,ff,fpl ); end
  #    if abs(ff) < TOL, break; end        # stopping criteria
  #    l_old = l;
  #    if k>2 && ( abs(ff) > 10*abs(oldff+100) ) #|| abs(d) > 1e13 )
  #        l = 0; alpha = 1/2;  
  ##         oldff = f(0);
  #        oldff = sum(b2); oldff = oldff - epsilon2;
  #        if DISP, disp('restarting'); end
  #    else
  #        if alpha < 1, alpha = (alpha+1)/2; end
  #        l = l + alpha*d;
  #        oldff = ff;
  #        if l > 0
  #            l = 0;  # shouldn't be positive
  #            oldff = sum(b2);  oldff = oldff - epsilon2;
  #        end
  #    end
  #    if l_old == l && l == 0
  #        if DISP, disp('Making no progress; x = y is probably feasible'); end
  #        break;
  #    end
  #end
  ## if k == MAXIT && DEBUG, disp('maxed out iterations'); end
  #if l < 0
  #    xhat = -l*s.*b./( 1 - l*s2 );
  #    x = V*xhat + y;
  #else
  #    # y is already feasible, so no need to project
  #    l = 0;
  #    x = y;
  #end
  
  # End of original Matab code
  #---------------------
  
  DEBUG = True;
  #if nargin < 8
  #    DISP = false;
  #end
  # -- Parameters for Newton's method --
  MAXIT = 70;
  TOL = 1e-8 * epsilon;
  # TOL = max( TOL, 10*eps );  # we can't do better than machine precision
  
  #m = size(U,1);
  #n = size(V,1);
  m = U.shape[0]
  n = V.shape[0]
  mn = min(m,n);
  #if numel(S) > mn^2, S = diag(diag(S)); end  # S should be a small square matrix
  if S.size > mn**2:
    S = numpy.diag(numpy.diag(S))
  #r = size(S);
  r = S.shape[0] # S is square
  #if size(U,2) > r, U = U(:,1:r); end
  if U.shape[1] > r:
    U = U[:,:r]
  #if size(V,2) > r, V = V(:,1:r); end
  if V.shape[1] > r:
    V = V[:,:r]
  
  s = numpy.diag(S);
  s2 = s**2;
  
  # What we want to do:
  #   b = b - A*y;
  #   bb = U'*b;
  
  # if A doesn't have full row rank, then b may not be in the range
  #if size(U,1) > size(U,2)
  if U.shape[0] > U.shape[1]:
      #bRange = U*(U'*b);
      bRange = numpy.dot(U, numpy.dot(U.T, b))
      bNull = b - bRange;
      epsilon = math.sqrt( epsilon**2 - numpy.linalg.norm(bNull)**2 );
  #end
  #b = U'*b - S*(V'*y);  # parenthesis is very important!  This is expensive.
  b = numpy.dot(U.T,b) - numpy.dot(S, numpy.dot(V.T,y))
      
  # b2 = b.^2;
  b2 = abs(b)**2;  # for complex data
  bs2 = b2*s2;
  epsilon2 = epsilon**2;
  
  # The following routine need to be fast
  # For efficiency (at cost of transparency), we are writing the calculations
  # in a way that minimize number of operations.  The functions "f"
  # and "fp" represent f and its derivative.
  
  # f = @(lambda) sum( b2 .*(1-lambda*s2).^(-2) ) - epsilon^2;
  # fp = @(lambda) 2*sum( bs2 .*(1-lambda*s2).^(-3) );
  
  #if nargin < 7, lambda0 = 0; end
  l = lambda0;
  oldff = 0;
  one = numpy.ones(m);
  alpha = 1;      # take full Newton steps
  #for k = 1:MAXIT
  for k in numpy.arange(MAXIT):
      # make f(l) and fp(l) as efficient as possible:
      #ls = one./(one-l*s2);
      ls = one/(one-l*s2)
      ls2 = ls**2;
      ls3 = ls2*ls;
      #ff = b2.'*ls2; # should be .', not ', even for complex data
      ff = numpy.dot(b2.conj(), ls2)
      ff = ff - epsilon2;
      #fpl = 2*( bs2.'*ls3 );  # should be .', not ', even for complex data
      fpl = 2 * numpy.dot(bs2.conj(),ls3)
      #     ff = f(l);    # this is a little slower
      #     fpl = fp(l);  # this is a little slower
      d = -ff/fpl;
      #      if DISP, fprintf('#2d, lambda is #5.2f, f(lambda) is #.2e, f''(lambda) is #.2e\n',k,l,ff,fpl ); end
      if DISP:
        print k,', lambda is ',l,', f(lambda) is ',ff,', f''(lambda) is',fpl
      #if abs(ff) < TOL, break; end        # stopping criteria
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
      #end
      if l_old == l and l == 0:
          #if DISP, disp('Making no progress; x = y is probably feasible'); end
          if DISP:
            print 'Making no progress; x = y is probably feasible'
          break;
      #end
  #end
  # if k == MAXIT && DEBUG, disp('maxed out iterations'); end
  if l < 0:
      #xhat = -l*s.*b./( 1 - l*s2 );
      xhat = numpy.dot(-l, s*b/( 1. - numpy.dot(l,s2) ) )
      #x = V*xhat + y;
      x = numpy.dot(V,xhat) + y
  else:
      # y is already feasible, so no need to project
      l = 0;
      x = y.copy();
  #end 
  return x,k,l