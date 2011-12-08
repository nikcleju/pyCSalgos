# -*- coding: utf-8 -*-
"""
Created on Thu Dec 08 10:56:56 2011

@author: ncleju
"""

import numpy
import numpy.linalg

from numpy.random import RandomState
rng = RandomState()

def Generate_Analysis_Operator(d, p):
  # generate random tight frame with equal column norms
  if p == d:
    T = rng.randn(d,d);
    [Omega, discard] = numpy.qr(T);
  else:
    Omega = rng.randn(p, d);
    T = numpy.zeros((p, d));
    tol = 1e-8;
    max_j = 200;
    j = 1;
    while (sum(sum(abs(T-Omega))) > numpy.dot(tol,numpy.dot(p,d)) and j < max_j):
        j = j + 1;
        T = Omega;
        [U, S, Vh] = numpy.linalg.svd(Omega);
        V = Vh.T
        #Omega = U * [eye(d); zeros(p-d,d)] * V';
        Omega2 = numpy.dot(numpy.dot(U, numpy.concatenate((numpy.eye(d), numpy.zeros((p-d,d))))), V.transpose())
        #Omega = diag(1./sqrt(diag(Omega*Omega')))*Omega;
        Omega = numpy.dot(numpy.diag(1.0 / numpy.sqrt(numpy.diag(numpy.dot(Omega2,Omega2.transpose())))), Omega2)
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
  LambdaMat = numpy.zeros((k,numvectors))
  x0 = numpy.zeros((d,numvectors))
  y = numpy.zeros((m,numvectors))
  realnoise = numpy.zeros((m,numvectors))
  
  M = rng.randn(m,d);
  
  #for i=1:numvectors
  for i in range(0,numvectors):
    
    # Generate signals
    #if strcmp(normstr,'l0')
    if normstr == 'l0':
        # Unchanged
        
        #Lambda=randperm(p); 
        Lambda = rng.permutation(int(p));
        Lambda = numpy.sort(Lambda[0:k]); 
        LambdaMat[:,i] = Lambda; # store for output
        
        # The signal is drawn at random from the null-space defined by the rows 
        # of the matreix Omega(Lambda,:)
        [U,D,Vh] = numpy.linalg.svd(Omega[Lambda,:]);
        V = Vh.T
        NullSpace = V[:,k:];
        #print numpy.dot(NullSpace, rng.randn(d-k,1)).shape
        #print x0[:,i].shape
        x0[:,i] = numpy.squeeze(numpy.dot(NullSpace, rng.randn(d-k,1)));
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
    y[:,i]  = numpy.dot(M, x0[:,i])

    # Add noise
    t_norm = numpy.linalg.norm(y[:,i],2)
    n = numpy.squeeze(rng.randn(m, 1))
    # In case n i just a number, nuit an array, norm() fails
    if n.ndim == 0:
      nnorm = abs(n)
    else:
      nnorm = numpy.linalg.norm(n, 2);
    y[:,i] = y[:,i] + noiselevel * t_norm * n / nnorm
    realnoise[:,i] = noiselevel * t_norm * n / nnorm
  #end

  return x0,y,M,LambdaMat,realnoise
