# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 14:05:22 2011

@author: ncleju
"""

#from numpy import *
#from scipy import *
import numpy as np
import scipy as sp
from scipy import linalg
import math

from numpy.random import RandomState
rng = RandomState()



def Generate_Analysis_Operator(d, p):
  # generate random tight frame with equal column norms
  if p == d:
    T = rng.randn(d,d);
    [Omega, discard] = np.qr(T);
  else:
    Omega = rng.randn(p, d);
    T = np.zeros((p, d));
    tol = 1e-8;
    max_j = 200;
    j = 1;
    while (sum(sum(abs(T-Omega))) > np.dot(tol,np.dot(p,d)) and j < max_j):
        j = j + 1;
        T = Omega;
        [U, S, Vh] = sp.linalg.svd(Omega);
        V = Vh.T
        #Omega = U * [eye(d); zeros(p-d,d)] * V';
        Omega2 = np.dot(np.dot(U, np.concatenate((np.eye(d), np.zeros((p-d,d))))), V.transpose())
        #Omega = diag(1./sqrt(diag(Omega*Omega')))*Omega;
        Omega = np.dot(np.diag(1 / np.sqrt(np.diag(np.dot(Omega2,Omega2.transpose())))), Omega2)
    #end
    ##disp(j);
#end
  return Omega


def Generate_Data_Known_Omega(Omega, d,p,m,k,noiselevel, numvectors, normstr):
  #function [x0,y,M,LambdaMat] = Generate_Data_Known_Omega(Omega, d,p,m,k,noiselevel, numvectors, normstr)
  
  # Building an analysis problem, which includes the ingredients: 
  #   - Omega - the analysis operator of size p*d
  #   - M is anunderdetermined measurement matrix of size m*d (m<d)
  #   - x0 is a vector of length d that satisfies ||Omega*x0||=p-k
  #   - Lambda is the true location of these k zeros in Omega*x0
  #   - a measurement vector y0=Mx0 is computed
  #   - noise contaminated measurement vector y is obtained by
  #     y = y0 + n where n is an additive gaussian noise with norm(n,2)/norm(y0,2) = noiselevel
  # Added by Nic:
  #   - Omega = analysis operator
  #   - normstr: if 'l0', generate l0 sparse vector (unchanged). If 'l1',
  #   generate a vector of Laplacian random variables (gamma) and
  #   pseudoinvert to find x

  # Omega is known as input parameter
  #Omega=Generate_Analysis_Operator(d, p);
  # Omega = randn(p,d);
  # for i = 1:size(Omega,1)
  #     Omega(i,:) = Omega(i,:) / norm(Omega(i,:));
  # end
  
  #Init
  LambdaMat = np.zeros((k,numvectors))
  x0 = np.zeros((d,numvectors))
  y = np.zeros((m,numvectors))
  
  M = rng.randn(m,d);
  
  #for i=1:numvectors
  for i in range(0,numvectors):
    
    # Generate signals
    #if strcmp(normstr,'l0')
    if normstr == 'l0':
        # Unchanged
        
        #Lambda=randperm(p); 
        Lambda = rng.permutation(int(p));
        Lambda = np.sort(Lambda[0:k]); 
        LambdaMat[:,i] = Lambda; # store for output
        
        # The signal is drawn at random from the null-space defined by the rows 
        # of the matreix Omega(Lambda,:)
        [U,D,Vh] = sp.linalg.svd(Omega[Lambda,:]);
        V = Vh.T
        NullSpace = V[:,k:];
        #print np.dot(NullSpace, rng.randn(d-k,1)).shape
        #print x0[:,i].shape
        x0[:,i] = np.squeeze(np.dot(NullSpace, rng.randn(d-k,1)));
        # Nic: add orthogonality noise
        #     orthonoiseSNRdb = 6;
        #     n = randn(p,1);
        #     #x0(:,i) = x0(:,i) + n / norm(n)^2 * norm(x0(:,i))^2 / 10^(orthonoiseSNRdb/10);
        #     n = n / norm(n)^2 * norm(Omega * x0(:,i))^2 / 10^(orthonoiseSNRdb/10);
        #     x0(:,i) = pinv(Omega) * (Omega * x0(:,i) + n);
        
    #elseif strcmp(normstr, 'l1')
    elif normstr == 'l1':
        print('Nic says: not implemented yet')
        raise Exception('Nic says: not implemented yet')
        #gamma = laprnd(p,1,0,1);
        #x0(:,i) = Omega \ gamma;
    else:
        #error('normstr must be l0 or l1!');
        print('Nic says: not implemented yet')
        raise Exception('Nic says: not implemented yet')
    #end
    
    # Acquire measurements
    y[:,i]  = np.dot(M, x0[:,i])

    # Add noise
    t_norm = np.linalg.norm(y[:,i],2);
    n = np.squeeze(rng.randn(m, 1));
    y[:,i] = y[:,i] + noiselevel * t_norm * n / np.linalg.norm(n, 2);
  #end

  return x0,y,M,LambdaMat

#####################

#function [xhat, arepr, lagmult] = ArgminOperL2Constrained(y, M, MH, Omega, OmegaH, Lambdahat, xinit, ilagmult, params)
def ArgminOperL2Constrained(y, M, MH, Omega, OmegaH, Lambdahat, xinit, ilagmult, params):
  
    #
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
    #
    
    #d = length(xinit)
    d = xinit.size
    lagmultmax = 1e5;
    lagmultmin = 1e-4;
    lagmultfactor = 2;
    accuracy_adjustment_exponent = 4/5;
    lagmult = max(min(ilagmult, lagmultmax), lagmultmin);
    was_infeasible = 0;
    was_feasible = 0;
    
    #######################################################################
    ## Computation done using direct matrix computation from matlab. (no conjugate gradient method.)
    #######################################################################
    #if strcmp(params.l2solver, 'pseudoinverse')
    if params['solver'] == 'pseudoinverse':
    #if strcmp(class(M), 'double') && strcmp(class(Omega), 'double')
      if M.dtype == 'float64' and Omega.dtype == 'double':
        while 1:
            alpha = math.sqrt(lagmult);
            #xhat = np.concatenate((M, alpha*Omega(Lambdahat,:)]\[y; zeros(length(Lambdahat), 1)];
            xhat = np.concatenate((M, np.linalg.lstsq(alpha*Omega[Lambdahat,:],np.concatenate((y, np.zeros(Lambdahat.size, 1))))));
            temp = np.linalg.norm(y - np.dot(M,xhat), 2);
            #disp(['fidelity error=', num2str(temp), ' lagmult=', num2str(lagmult)]);
            if temp <= params['noise_level']:
                was_feasible = 1;
                if was_infeasible == 1:
                    break;
                else:
                    lagmult = lagmult*lagmultfactor;
            elif temp > params['noise_level']:
                was_infeasible = 1;
                if was_feasible == 1:
                    xhat = xprev;
                    break;
                lagmult = lagmult/lagmultfactor;
            if lagmult < lagmultmin or lagmult > lagmultmax:
                break;
            xprev = xhat;
        arepr = np.dot(Omega[Lambdahat, :], xhat);
        return xhat,arepr,lagmult;


    ########################################################################
    ## Computation using conjugate gradient method.
    ########################################################################
    #if strcmp(class(MH),'function_handle') 
    if hasattr(MH, '__call__'):
        b = MH(y);
    else:
        b = np.dot(MH, y);
    
    norm_b = np.linalg.norm(b, 2);
    xhat = xinit;
    xprev = xinit;
    residual = TheHermitianMatrix(xhat, M, MH, Omega, OmegaH, Lambdahat, lagmult) - b;
    direction = -residual;
    iter = 0;
    
    while iter < params.max_inner_iteration:
        iter = iter + 1;
        alpha = np.linalg.norm(residual,2)**2 / np.dot(direction.T, TheHermitianMatrix(direction, M, MH, Omega, OmegaH, Lambdahat, lagmult));
        xhat = xhat + alpha*direction;
        prev_residual = residual;
        residual = TheHermitianMatrix(xhat, M, MH, Omega, OmegaH, Lambdahat, lagmult) - b;
        beta = np.linalg.norm(residual,2)**2 / np.linalg.norm(prev_residual,2)**2;
        direction = -residual + beta*direction;
        
        if np.linalg.norm(residual,2)/norm_b < params['l2_accuracy']*(lagmult**(accuracy_adjustment_exponent)) or iter == params['max_inner_iteration']:
            #if strcmp(class(M), 'function_handle')
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
            
            
            #disp(['residual=', num2str(norm(residual,2)), ' norm_b=', num2str(norm_b), ' omegapart=', num2str(u), ' fidelity error=', num2str(temp), ' lagmult=', num2str(lagmult), ' iter=', num2str(iter)]);
            
            if temp <= params['noise_level']:
                was_feasible = 1;
                if was_infeasible == 1:
                    break;
                else:
                    lagmult = lagmultfactor*lagmult;
                    residual = TheHermitianMatrix(xhat, M, MH, Omega, OmegaH, Lambdahat, lagmult) - b;
                    direction = -residual;
                    iter = 0;
            elif temp > params['noise_level']:
                lagmult = lagmult/lagmultfactor;
                if was_feasible == 1:
                    xhat = xprev;
                    break;
                was_infeasible = 1;
                residual = TheHermitianMatrix(xhat, M, MH, Omega, OmegaH, Lambdahat, lagmult) - b;
                direction = -residual;
                iter = 0;
            if lagmult > lagmultmax or lagmult < lagmultmin:
                break;
            xprev = xhat;
        #elseif norm(xprev-xhat)/norm(xhat) < 1e-2
        #    disp(['rel_change=', num2str(norm(xprev-xhat)/norm(xhat))]);
        #    if strcmp(class(M), 'function_handle')
        #        temp = norm(y-M(xhat), 2);
        #    else
        #        temp = norm(y-M*xhat, 2);
        #    end
    #
    #        if temp > 1.2*params.noise_level
    #            was_infeasible = 1;
    #            lagmult = lagmult/lagmultfactor;
    #            xprev = xhat;
    #        end
    
    #disp(['fidelity_error=', num2str(temp)]);
    print 'fidelity_error=',temp
    #if iter == params['max_inner_iteration']:
        #disp('max_inner_iteration reached. l2_accuracy not achieved.');
    
    ##
    # Compute analysis representation for xhat
    ##
    #if strcmp(class(Omega),'function_handle') 
    if hasattr(Omega, '__call__'):
        temp = Omega(xhat);
        arepr = temp(Lambdahat);
    else:    ## here Omega is assumed to be a matrix
        arepr = np.dot(Omega[Lambdahat, :], xhat);
    
    return xhat,arepr,lagmult


##
# This function computes (M'*M + lm*Omega(L,:)'*Omega(L,:)) * x.
##
#function w = TheHermitianMatrix(x, M, MH, Omega, OmegaH, L, lm)
def TheHermitianMatrix(x, M, MH, Omega, OmegaH, L, lm):
    #if strcmp(class(M), 'function_handle')
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
