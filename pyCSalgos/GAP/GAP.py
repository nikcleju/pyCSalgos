# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 14:05:22 2011

@author: ncleju
"""


import numpy
import numpy.linalg
import scipy as sp
import scipy.stats

import math


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
    lagmultfactor = 2.0;
    accuracy_adjustment_exponent = 4/5.;
    lagmult = max(min(ilagmult, lagmultmax), lagmultmin);
    was_infeasible = 0;
    was_feasible = 0;
    
    #######################################################################
    ## Computation done using direct matrix computation from matlab. (no conjugate gradient method.)
    #######################################################################
    #if strcmp(params.l2solver, 'pseudoinverse')
    if params['l2solver'] == 'pseudoinverse':
    #if strcmp(class(M), 'double') && strcmp(class(Omega), 'double')
      #if M.dtype == 'float64' and Omega.dtype == 'double':
      if 1:
        while True:
            alpha = math.sqrt(lagmult);
            xhat = numpy.linalg.lstsq(numpy.concatenate((M, alpha*Omega[Lambdahat,:])), numpy.concatenate((y, numpy.zeros(Lambdahat.size))))[0]
            temp = numpy.linalg.norm(y - numpy.dot(M,xhat), 2);
            #disp(['fidelity error=', num2str(temp), ' lagmult=', num2str(lagmult)]);
            if temp <= params['noise_level']:
                was_feasible = True;
                if was_infeasible:
                    break;
                else:
                    lagmult = lagmult*lagmultfactor;
            elif temp > params['noise_level']:
                was_infeasible = True;
                if was_feasible:
                    xhat = xprev.copy();
                    break;
                lagmult = lagmult/lagmultfactor;
            if lagmult < lagmultmin or lagmult > lagmultmax:
                break;
            xprev = xhat.copy();
        arepr = numpy.dot(Omega[Lambdahat, :], xhat);
        return xhat,arepr,lagmult;


    ########################################################################
    ## Computation using conjugate gradient method.
    ########################################################################
    #if strcmp(class(MH),'function_handle') 
    if hasattr(MH, '__call__'):
        b = MH(y);
    else:
        b = numpy.dot(MH, y);
    
    norm_b = numpy.linalg.norm(b, 2);
    xhat = xinit.copy();
    xprev = xinit.copy();
    residual = TheHermitianMatrix(xhat, M, MH, Omega, OmegaH, Lambdahat, lagmult) - b;
    direction = -residual;
    iter = 0;
    
    while iter < params.max_inner_iteration:
        iter = iter + 1;
        alpha = numpy.linalg.norm(residual,2)**2 / numpy.dot(direction.T, TheHermitianMatrix(direction, M, MH, Omega, OmegaH, Lambdahat, lagmult));
        xhat = xhat + alpha*direction;
        prev_residual = residual.copy();
        residual = TheHermitianMatrix(xhat, M, MH, Omega, OmegaH, Lambdahat, lagmult) - b;
        beta = numpy.linalg.norm(residual,2)**2 / numpy.linalg.norm(prev_residual,2)**2;
        direction = -residual + beta*direction;
        
        if numpy.linalg.norm(residual,2)/norm_b < params['l2_accuracy']*(lagmult**(accuracy_adjustment_exponent)) or iter == params['max_inner_iteration']:
            #if strcmp(class(M), 'function_handle')
            if hasattr(M, '__call__'):
                temp = numpy.linalg.norm(y-M(xhat), 2);
            else:
                temp = numpy.linalg.norm(y-numpy.dot(M,xhat), 2);
            
            #if strcmp(class(Omega), 'function_handle')
            if hasattr(Omega, '__call__'):
                u = Omega(xhat);
                u = math.sqrt(lagmult)*numpy.linalg.norm(u(Lambdahat), 2);
            else:
                u = math.sqrt(lagmult)*numpy.linalg.norm(Omega[Lambdahat,:]*xhat, 2);
            
            
            #disp(['residual=', num2str(norm(residual,2)), ' norm_b=', num2str(norm_b), ' omegapart=', num2str(u), ' fidelity error=', num2str(temp), ' lagmult=', num2str(lagmult), ' iter=', num2str(iter)]);
            
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
        arepr = numpy.dot(Omega[Lambdahat, :], xhat);
    
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
        w = numpy.dot(numpy.dot(MH, M), x);
    
    if hasattr(Omega, '__call__'):
        v = Omega(x);
        vt = numpy.zeros(v.size);
        vt[L] = v[L].copy();
        w = w + lm*OmegaH(vt);
    else:    ## Omega is assumed to be a matrix and OmegaH is its conjugate transpose
        w = w + lm*numpy.dot(numpy.dot(OmegaH[:, L],Omega[L, :]),x);
    
    return w

def GAP(y, M, MH, Omega, OmegaH, params, xinit):
  #function [xhat, Lambdahat] = GAP(y, M, MH, Omega, OmegaH, params, xinit)
  
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
  
  #d = length(xinit(:));
  d = xinit.size
  
  #if strcmp(class(Omega), 'function_handle')
  #    p = length(Omega(zeros(d,1)));
  #else    ## Omega is a matrix
  #    p = size(Omega, 1);
  #end
  if hasattr(Omega, '__call__'):
      p = Omega(numpy.zeros((d,1)))
  else:
      p = Omega.shape[0]
  
  
  iter = 0
  lagmult = 1e-4
  #Lambdahat = 1:p;
  Lambdahat = numpy.arange(p)
  #while iter < params.num_iteration
  while iter < params["num_iteration"]:
      iter = iter + 1
      #[xhat, analysis_repr, lagmult] = ArgminOperL2Constrained(y, M, MH, Omega, OmegaH, Lambdahat, xinit, lagmult, params);
      xhat,analysis_repr,lagmult = ArgminOperL2Constrained(y, M, MH, Omega, OmegaH, Lambdahat, xinit, lagmult, params)
      #[to_be_removed, maxcoef] = FindRowsToRemove(analysis_repr, params.greedy_level);
      to_be_removed,maxcoef = FindRowsToRemove(analysis_repr, params["greedy_level"])
      #disp(['** maxcoef=', num2str(maxcoef), ' target=', num2str(params.stopping_coefficient_size), ' rows_remaining=', num2str(length(Lambdahat)), ' lagmult=', num2str(lagmult)]);
      #print '** maxcoef=',maxcoef,' target=',params['stopping_coefficient_size'],' rows_remaining=',Lambdahat.size,' lagmult=',lagmult
      if check_stopping_criteria(xhat, xinit, maxcoef, lagmult, Lambdahat, params):
          break

      xinit = xhat.copy()
      #Lambdahat[to_be_removed] = []
      Lambdahat = numpy.delete(Lambdahat.squeeze(),to_be_removed)
  
      #n = sqrt(d);
      #figure(9);
      #RR = zeros(2*n, n-1);
      #RR(Lambdahat) = 1;
      #XD = ones(n, n);
      #XD(:, 2:end) = XD(:, 2:end) .* RR(1:n, :);
      #XD(:, 1:(end-1)) = XD(:, 1:(end-1)) .* RR(1:n, :);
      #XD(2:end, :) = XD(2:end, :) .* RR((n+1):(2*n), :)';
      #XD(1:(end-1), :) = XD(1:(end-1), :) .* RR((n+1):(2*n), :)';
      #XD = FD2DiagnosisPlot(n, Lambdahat);
      #imshow(XD);
      #figure(10);
      #imshow(reshape(real(xhat), n, n));
  
      # Added by Nic: if Lambdahat is empty here, return
      if Lambdahat.size == 0:
        break;
        
  #return;
  return xhat,Lambdahat
 
def FindRowsToRemove(analysis_repr, greedy_level):
#function [to_be_removed, maxcoef] = FindRowsToRemove(analysis_repr, greedy_level)

    #abscoef = abs(analysis_repr(:));
    abscoef = numpy.abs(analysis_repr)
    #n = length(abscoef);
    n = abscoef.size
    #maxcoef = max(abscoef);
    maxcoef = abscoef.max()
    if greedy_level >= 1:
        #qq = quantile(abscoef, 1.0-greedy_level/n);
        #qq = sp.stats.mstats.mquantiles(abscoef, 1.0-greedy_level/n, 0.5, 0.5)        
        qq = sp.stats.mstats.mquantiles(abscoef, 1.0-greedy_level/n)
    else:
        qq = maxcoef*greedy_level

    #to_be_removed = find(abscoef >= qq);
    # [0] needed because nonzero() returns a tuple of arrays!
    to_be_removed = numpy.nonzero(abscoef >= qq)[0]
    #return;
    return to_be_removed,maxcoef

def check_stopping_criteria(xhat, xinit, maxcoef, lagmult, Lambdahat, params):
#function r = check_stopping_criteria(xhat, xinit, maxcoef, lagmult, Lambdahat, params)

    #if isfield(params, 'stopping_coefficient_size') && maxcoef < params.stopping_coefficient_size
    if ('stopping_coefficient_size' in params) and maxcoef < params['stopping_coefficient_size']:
        return 1

    #if isfield(params, 'stopping_lagrange_multiplier_size') && lagmult > params.stopping_lagrange_multiplier_size
    if ('stopping_lagrange_multiplier_size' in params) and lagmult > params['stopping_lagrange_multiplier_size']:
        return 1

    #if isfield(params, 'stopping_relative_solution_change') && norm(xhat-xinit)/norm(xhat) < params.stopping_relative_solution_change
    if ('stopping_relative_solution_change' in params) and numpy.linalg.norm(xhat-xinit)/numpy.linalg.norm(xhat) < params['stopping_relative_solution_change']:
        return 1

    #if isfield(params, 'stopping_cosparsity') && length(Lambdahat) < params.stopping_cosparsity
    if ('stopping_cosparsity' in params) and Lambdahat.size < params['stopping_cosparsity']:
        return 1
    
    return 0