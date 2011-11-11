import numpy as np
import scipy.linalg
import time
import math


#function [s, err_mse, iter_time]=greed_omp_qr(x,A,m,varargin)
def greed_omp_qr(x,A,m,opts=[]):
# greed_omp_qr: Orthogonal Matching Pursuit algorithm based on QR
# factorisation
# Nic: translated to Python on 19.10.2011. Original Matlab Code by Thomas Blumensath
###########################################################################
# Usage
# [s, err_mse, iter_time]=greed_omp_qr(x,P,m,'option_name','option_value')
###########################################################################
###########################################################################
# Input
#   Mandatory:
#               x   Observation vector to be decomposed
#               P   Either:
#                       1) An nxm matrix (n must be dimension of x)
#                       2) A function handle (type "help function_format" 
#                          for more information)
#                          Also requires specification of P_trans option.
#                       3) An object handle (type "help object_format" for 
#                          more information)
#               m   length of s 
#
#   Possible additional options:
#   (specify as many as you want using 'option_name','option_value' pairs)
#   See below for explanation of options:
#__________________________________________________________________________
#   option_name    |     available option_values                | default
#--------------------------------------------------------------------------
#   stopCrit       | M, corr, mse, mse_change                   | M
#   stopTol        | number (see below)                         | n/4
#   P_trans        | function_handle (see below)                | 
#   maxIter        | positive integer (see below)               | n
#   verbose        | true, false                                | false
#   start_val      | vector of length m                         | zeros
#
#   Available stopping criteria :
#               M           -   Extracts exactly M = stopTol elements.
#               corr        -   Stops when maximum correlation between
#                               residual and atoms is below stopTol value.
#               mse         -   Stops when mean squared error of residual 
#                               is below stopTol value.
#               mse_change  -   Stops when the change in the mean squared 
#                               error falls below stopTol value.
#
#   stopTol: Value for stopping criterion.
#
#   P_trans: If P is a function handle, then P_trans has to be specified and 
#            must be a function handle. 
#
#   maxIter: Maximum number of allowed iterations.
#
#   verbose: Logical value to allow algorithm progress to be displayed.
#
#   start_val: Allows algorithms to start from partial solution.
#
###########################################################################
# Outputs
#    s              Solution vector 
#    err_mse        Vector containing mse of approximation error for each 
#                   iteration
#    iter_time      Vector containing computation times for each iteration
#
###########################################################################
# Description
#   greed_omp_qr performs a greedy signal decomposition. 
#   In each iteration a new element is selected depending on the inner
#   product between the current residual and columns in P.
#   The non-zero elements of s are approximated by orthogonally projecting 
#   x onto the selected elements in each iteration.
#   The algorithm uses QR decomposition.
#
# See Also
#   greed_omp_chol, greed_omp_cg, greed_omp_cgp, greed_omp_pinv, 
#   greed_omp_linsolve, greed_gp, greed_nomp
#
# Copyright (c) 2007 Thomas Blumensath
#
# The University of Edinburgh
# Email: thomas.blumensath@ed.ac.uk
# Comments and bug reports welcome
#
# This file is part of sparsity Version 0.1
# Created: April 2007
#
# Part of this toolbox was developed with the support of EPSRC Grant
# D000246/1
#
# Please read COPYRIGHT.m for terms and conditions.

    ###########################################################################
    #                    Default values and initialisation
    ###########################################################################
    #[n1 n2]=size(x);
    #n1,n2 = x.shape
    #if n2 == 1
    #    n=n1;
    #elseif n1 == 1
    #    x=x';
    #    n=n2;
    #else
    #   display('x must be a vector.');
    #   return
    #end
    if x.ndim != 1:
      print 'x must be a vector.'
      return
    n = x.size
        
    #sigsize     = x'*x/n;
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
    #switch nargout 
    #    case 3
    #        comp_err=true;
    #        comp_time=true;
    #    case 2 
    #        comp_err=true;
    #        comp_time=false;
    #    case 1
    #        comp_err=false;
    #        comp_time=false;
    #    case 0
    #        error('Please assign output variable.')
    #    otherwise
    #        error('Too many output arguments specified')
    #end
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
    # Put option into nice format
    #Options={};
    #OS=nargin-3;
    #c=1;
    #for i=1:OS
    #    if isa(varargin{i},'cell')
    #        CellSize=length(varargin{i});
    #        ThisCell=varargin{i};
    #        for j=1:CellSize
    #            Options{c}=ThisCell{j};
    #            c=c+1;
    #        end
    #    else
    #        Options{c}=varargin{i};
    #        c=c+1;
    #    end
    #end
    #OS=length(Options);
    #if rem(OS,2)
    #   error('Something is wrong with argument name and argument value pairs.') 
    #end
    #
    #for i=1:2:OS
    #   switch Options{i}
    #        case {'stopCrit'}
    #            if (strmatch(Options{i+1},{'M'; 'corr'; 'mse'; 'mse_change'},'exact'));
    #                STOPCRIT    = Options{i+1};  
    #            else error('stopCrit must be char string [M, corr, mse, mse_change]. Exiting.'); end 
    #        case {'stopTol'}
    #            if isa(Options{i+1},'numeric') ; STOPTOL     = Options{i+1};   
    #            else error('stopTol must be number. Exiting.'); end
    #        case {'P_trans'} 
    #            if isa(Options{i+1},'function_handle'); Pt = Options{i+1};   
    #            else error('P_trans must be function _handle. Exiting.'); end
    #        case {'maxIter'}
    #            if isa(Options{i+1},'numeric'); MAXITER     = Options{i+1};             
    #            else error('maxIter must be a number. Exiting.'); end
    #        case {'verbose'}
    #            if isa(Options{i+1},'logical'); verbose     = Options{i+1};   
    #            else error('verbose must be a logical. Exiting.'); end 
    #        case {'start_val'}
    #            if isa(Options{i+1},'numeric') & length(Options{i+1}) == m ;
    #                s_initial     = Options{i+1};   
    #                initial_given=1;
    #            else error('start_val must be a vector of length m. Exiting.'); end
    #        otherwise
    #            error('Unrecognised option. Exiting.') 
    #   end
    #end
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

    #if strcmp(STOPCRIT,'M') 
    #    maxM=STOPTOL;
    #else
    #    maxM=MAXITER;
    #end
    if STOPCRIT == 'M':
        maxM = STOPTOL
    else:
        maxM = MAXITER
    
    #    if nargout >=2
    #        err_mse = zeros(maxM,1);
    #    end
    #    if nargout ==3
    #        iter_time = zeros(maxM,1);
    #    end
    if opts['nargout'] >= 2:
        err_mse = np.zeros(maxM)
    if opts['nargout'] == 3:
        iter_time = np.zeros(maxM)
    
    ###########################################################################
    #                        Make P and Pt functions
    ###########################################################################
    #if          isa(A,'float')      P =@(z) A*z;  Pt =@(z) A'*z;
    #elseif      isobject(A)         P =@(z) A*z;  Pt =@(z) A'*z;
    #elseif      isa(A,'function_handle') 
    #    try
    #        if          isa(Pt,'function_handle'); P=A;
    #        else        error('If P is a function handle, Pt also needs to be a function handle. Exiting.'); end
    #    catch error('If P is a function handle, Pt needs to be specified. Exiting.'); end
    #else        error('P is of unsupported type. Use matrix, function_handle or object. Exiting.'); end
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
    #    mask=zeros(m,1);
    #    mask(ceil(rand*m))=1;
    #    nP=norm(P(mask));
    #    if abs(1-nP)>1e-3;
    #        display('Dictionary appears not to have unit norm columns.')
    #    end
    mask = np.zeros(m)
    mask[math.floor(np.random.rand() * m)] = 1
    #nP = np.linalg.norm(P(mask))
    #if abs(1-nP) > 1e-3:
    #    print 'Dictionary appears not to have unit norm columns.'
    
    ###########################################################################
    #              Check if we have enough memory and initialise 
    ###########################################################################
    #        try Q=zeros(n,maxM);
    #        catch error('Variable size is too large. Please try greed_omp_chol algorithm or reduce MAXITER.')
    #        end 
    #        try R=zeros(maxM);
    #        catch error('Variable size is too large. Please try greed_omp_chol algorithm or reduce MAXITER.')
    #        end
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
    #if initial_given ==1;
    #    IN          = find(s_initial);
    #    if ~isempty(IN)
    #        Residual    = x-P(s_initial);
    #        lengthIN=length(IN);
    #        z=[];
    #        for k=1:length(IN)
    #            # Extract new element
    #             mask=zeros(m,1);
    #             mask(IN(k))=1;
    #             new_element=P(mask);
    #
    #            # Orthogonalise new element 
    #             qP=Q(:,1:k-1)'*new_element;
    #             q=new_element-Q(:,1:k-1)*(qP);
    #
    #             nq=norm(q);
    #             q=q/nq;
    #            # Update QR factorisation 
    #             R(1:k-1,k)=qP;
    #             R(k,k)=nq;
    #             Q(:,k)=q;
    #
    #             z(k)=q'*x;
    #        end
    #        s           = s_initial;
    #        Residual=x-Q(:,k)*z;
    #        oldERR      = Residual'*Residual/n;
    #    else
    #    	IN          = [];
    #        Residual    = x;
    #        s           = s_initial;
    #        sigsize     = x'*x/n;
    #        oldERR      = sigsize;
    #        k=0;
    #        z=[];
    #    end
    #    
    #else
    #    IN          = [];
    #    Residual    = x;
    #    s           = s_initial;
    #    sigsize     = x'*x/n;
    #    oldERR      = sigsize;
    #    k=0;
    #    z=[];
    #end
    if initial_given == 1:
        #IN = find(s_initial);
        IN = np.nonzero(s_initial)[0].tolist()
        #if ~isempty(IN)
        if IN.size > 0:
            Residual = x - P(s_initial)
            lengthIN = IN.size
            z = np.array([])
            #for k=1:length(IN)
            for k in np.arange(IN.size):
                # Extract new element
                 mask = np.zeros(m)
                 mask[IN[k]] = 1
                 new_element = P(mask)
                 
                 # Orthogonalise new element 
                 #qP=Q(:,1:k-1)'*new_element;
                 if k-1 >= 0:
                     qP = np.dot(Q[:,0:k].T , new_element)
                     #q=new_element-Q(:,1:k-1)*(qP);
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
            #end
            s        = s_initial.copy()
            Residual = x - np.dot(Q[:,k] , z)
            oldERR   = np.vdot(Residual , Residual) / n;
        else:
            #IN          = np.array([], dtype = int)
            IN          = np.array([], dtype = int).tolist()
            Residual    = x.copy()
            s           = s_initial.copy()
            sigsize     = np.vdot(x , x) / n
            oldERR      = sigsize
            k = 0
            #z = np.array([])
            z = []
        #end
        
    else:
        #IN          = np.array([], dtype = int)
        IN          = np.array([], dtype = int).tolist()
        Residual    = x.copy()
        s           = s_initial.copy()
        sigsize     = np.vdot(x , x) / n
        oldERR      = sigsize
        k = 0
        #z = np.array([])
        z = []
    #end
    
    ###########################################################################
    #                        Main algorithm
    ###########################################################################
    #    if verbose
    #       display('Main iterations...') 
    #    end
    #    tic
    #    t=0;
    #    DR=Pt(Residual);
    #    done = 0;
    #    iter=1;
    if verbose:
       print 'Main iterations...'
    tic = time.time()
    t = 0
    DR = Pt(Residual)
    done = 0
    iter = 1
    
    #while ~done
    #    
    #     # Select new element
    #     DR(IN)=0;
    #     # Nic: replace selection with random variable
    #     # i.e. Randomized OMP!!
    #     # DON'T FORGET ABOUT THIS!!
    #     [v I]=max(abs(DR));
    #     #I = randp(exp(abs(DR).^2 ./ (norms.^2)'), [1 1]);
    #     IN=[IN I];
    #
    #    
    #     k=k+1;
    #     # Extract new element
    #     mask=zeros(m,1);
    #     mask(IN(k))=1;
    #     new_element=P(mask);
    #
    #    # Orthogonalise new element 
    #     qP=Q(:,1:k-1)'*new_element;
    #     q=new_element-Q(:,1:k-1)*(qP);
    #
    #     nq=norm(q);
    #     q=q/nq;
    #    # Update QR factorisation 
    #     R(1:k-1,k)=qP;
    #     R(k,k)=nq;
    #     Q(:,k)=q;
    #
    #     z(k)=q'*x;
    #   
    #    # New residual 
    #     Residual=Residual-q*(z(k));
    #     DR=Pt(Residual);
    #     
    #     ERR=Residual'*Residual/n;
    #     if comp_err
    #         err_mse(iter)=ERR;
    #     end
    #     
    #     if comp_time
    #         iter_time(iter)=toc;
    #     end
    #
    ############################################################################
    ##                        Are we done yet?
    ############################################################################
    #     
    #     if strcmp(STOPCRIT,'M')
    #         if iter >= STOPTOL
    #             done =1;
    #         elseif verbose && toc-t>10
    #            display(sprintf('Iteration #i. --- #i iterations to go',iter ,STOPTOL-iter)) 
    #            t=toc;
    #         end
    #    elseif strcmp(STOPCRIT,'mse')
    #         if comp_err
    #            if err_mse(iter)<STOPTOL;
    #                done = 1; 
    #            elseif verbose && toc-t>10
    #                display(sprintf('Iteration #i. --- #i mse',iter ,err_mse(iter))) 
    #                t=toc;
    #            end
    #         else
    #             if ERR<STOPTOL;
    #                done = 1; 
    #             elseif verbose && toc-t>10
    #                display(sprintf('Iteration #i. --- #i mse',iter ,ERR)) 
    #                t=toc;
    #             end
    #         end
    #     elseif strcmp(STOPCRIT,'mse_change') && iter >=2
    #         if comp_err && iter >=2
    #              if ((err_mse(iter-1)-err_mse(iter))/sigsize <STOPTOL);
    #                done = 1; 
    #             elseif verbose && toc-t>10
    #                display(sprintf('Iteration #i. --- #i mse change',iter ,(err_mse(iter-1)-err_mse(iter))/sigsize )) 
    #                t=toc;
    #             end
    #         else
    #             if ((oldERR - ERR)/sigsize < STOPTOL);
    #                done = 1; 
    #             elseif verbose && toc-t>10
    #                display(sprintf('Iteration #i. --- #i mse change',iter ,(oldERR - ERR)/sigsize)) 
    #                t=toc;
    #             end
    #         end
    #     elseif strcmp(STOPCRIT,'corr') 
    #          if max(abs(DR)) < STOPTOL;
    #             done = 1; 
    #          elseif verbose && toc-t>10
    #                display(sprintf('Iteration #i. --- #i corr',iter ,max(abs(DR)))) 
    #                t=toc;
    #          end
    #     end
    #     
    #    # Also stop if residual gets too small or maxIter reached
    #     if comp_err
    #         if err_mse(iter)<1e-16
    #             display('Stopping. Exact signal representation found!')
    #             done=1;
    #         end
    #     else
    #
    #
    #         if iter>1
    #             if ERR<1e-16
    #                 display('Stopping. Exact signal representation found!')
    #                 done=1;
    #             end
    #         end
    #     end
    #
    #     if iter >= MAXITER
    #         display('Stopping. Maximum number of iterations reached!')
    #         done = 1; 
    #     end
    #     
    ############################################################################
    ##                    If not done, take another round
    ############################################################################
    #   
    #     if ~done
    #        iter=iter+1;
    #        oldERR=ERR;
    #     end
    #end
    while not done:
         
         # Select new element
         DR[IN]=0
         #[v I]=max(abs(DR));
         #v = np.abs(DR).max()
         I = np.abs(DR).argmax()
         #IN = np.concatenate((IN,I))
         IN.append(I)
         
         
         #k = k + 1  Move to end, since is zero based
         
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

         #z[k]=np.vdot(q , x)
         z.append(np.vdot(q , x))
       
         # New residual 
         Residual = Residual - q * (z[k])
         DR = Pt(Residual)
         
         ERR = np.vdot(Residual , Residual) / n
         if comp_err:
             err_mse[iter-1] = ERR
         #end
         
         if comp_time:
             iter_time[iter-1] = time.time() - tic
         #end
         
         ###########################################################################
         #                        Are we done yet?
         ###########################################################################
         if STOPCRIT == 'M':
             if iter >= STOPTOL:
                 done = 1
             elif verbose and time.time() - t > 10.0/1000: # time() returns sec
                #display(sprintf('Iteration #i. --- #i iterations to go',iter ,STOPTOL-iter)) 
                print 'Iteration '+iter+'. --- '+(STOPTOL-iter)+' iterations to go'
                t = time.time()
             #end
         elif STOPCRIT =='mse':
             if comp_err:
                if err_mse[iter-1] < STOPTOL:
                    done = 1 
                elif verbose and time.time() - t > 10.0/1000: # time() returns sec
                    #display(sprintf('Iteration #i. --- #i mse',iter ,err_mse(iter))) 
                    print 'Iteration '+iter+'. --- '+err_mse[iter-1]+' mse'
                    t = time.time()
                #end
             else:
                 if ERR < STOPTOL:
                    done = 1 
                 elif verbose and time.time() - t > 10.0/1000: # time() returns sec
                    #display(sprintf('Iteration #i. --- #i mse',iter ,ERR)) 
                    print 'Iteration '+iter+'. --- '+ERR+' mse'
                    t = time.time()
                 #end
             #end
         elif STOPCRIT == 'mse_change' and iter >=2:
             if comp_err and iter >=2:
                 if ((err_mse[iter-2] - err_mse[iter-1])/sigsize < STOPTOL):
                    done = 1
                 elif verbose and time.time() - t > 10.0/1000: # time() returns sec
                    #display(sprintf('Iteration #i. --- #i mse change',iter ,(err_mse(iter-1)-err_mse(iter))/sigsize )) 
                    print 'Iteration '+iter+'. --- '+((err_mse[iter-2]-err_mse[iter-1])/sigsize)+' mse change'
                    t = time.time()
                 #end
             else:
                 if ((oldERR - ERR)/sigsize < STOPTOL):
                    done = 1
                 elif verbose and time.time() - t > 10.0/1000: # time() returns sec
                    #display(sprintf('Iteration #i. --- #i mse change',iter ,(oldERR - ERR)/sigsize)) 
                    print 'Iteration '+iter+'. --- '+((oldERR - ERR)/sigsize)+' mse change'
                    t = time.time()
                 #end
             #end
         elif STOPCRIT == 'corr':
              if np.abs(DR).max() < STOPTOL:
                 done = 1 
              elif verbose and time.time() - t > 10.0/1000: # time() returns sec
                  #display(sprintf('Iteration #i. --- #i corr',iter ,max(abs(DR)))) 
                  print 'Iteration '+iter+'. --- '+(np.abs(DR).max())+' corr'
                  t = time.time()
              #end
          #end
         
         # Also stop if residual gets too small or maxIter reached
         if comp_err:
             if err_mse[iter-1] < 1e-14:
                 done = 1
                 # Nic: added verbose check
                 if verbose:
                     print 'Stopping. Exact signal representation found!'
             #end
         else:
             if iter > 1:
                 if ERR < 1e-14:
                     done = 1 
                     # Nic: added verbose check
                     if verbose:
                         print 'Stopping. Exact signal representation found!'
                 #end
             #end
         #end
         
         
         if iter >= MAXITER:
             done = 1 
             # Nic: added verbose check
             if verbose:
                 print 'Stopping. Maximum number of iterations reached!'
         #end
         
         ###########################################################################
         #                    If not done, take another round
         ###########################################################################
         if not done:
            iter = iter + 1
            oldERR = ERR
         #end
         
         # Moved here from front, since we are 0-based         
         k = k + 1
    #end
    
    ###########################################################################
    #            Now we can solve for s by back-substitution
    ###########################################################################
    #s(IN)=R(1:k,1:k)\z(1:k)';
    s[IN] = scipy.linalg.solve(R[0:k,0:k] , np.array(z[0:k]))
    
    ###########################################################################
    #                  Only return as many elements as iterations
    ###########################################################################
    if opts['nargout'] >= 2:
        err_mse = err_mse[0:iter-1]
    #end
    if opts['nargout'] == 3:
        iter_time = iter_time[0:iter-1]
    #end
    if verbose:
       print 'Done'
    #end
    
    # Return
    if opts['nargout'] == 1:
        return s
    elif opts['nargout'] == 2:
        return s, err_mse
    elif opts['nargout'] == 3:
        return s, err_mse, iter_time
    
    # Change history
    #
    # 8 of Februray: Algo does no longer stop if dictionary is not normaliesd. 

    # End of greed_omp_qr() function
    #--------------------------------
    
    
def omp_qr(x, dict, D, natom, tolerance):
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
    x_hat = np.zeros((dictsize,1))
    x_hat[gamma[0:k]] = scipy.linalg.solve_triangular(tempR,w)

    return x_hat, gamma