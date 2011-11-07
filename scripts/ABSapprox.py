# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 18:08:40 2011

@author: Nic
"""

import numpy as np
import pyCSalgos
import pyCSalgos.GAP.GAP
import pyCSalgos.SL0.SL0_approx

# Define functions that prepare arguments for each algorithm call
def gap_paramsetup(y,M,Omega,epsilon,lbd):
  gapparams = {"num_iteration" : 1000,\
                   "greedy_level" : 0.9,\
                   "stopping_coefficient_size" : 1e-4,\
                   "l2solver" : 'pseudoinverse',\
                   "noise_level": epsilon}
  return y,M,M.T,Omega,Omega.T,gapparams,np.zeros(Omega.shape[1])
def sl0_paramsetup(y,M,Omega,epsilon,lbd):
  
  N,n = Omega.shape
  D = np.linalg.pinv(Omega)
  U,S,Vt = np.linalg.svd(D)
  aggDupper = np.dot(M,D)
  aggDlower = Vt[-(N-n):,:]
  aggD = np.concatenate((aggDupper, lbd * aggDlower))
  aggy = np.concatenate((y, np.zeros(N-n)))
  
  sigmamin = 0.1
  return aggD,aggy,epsilon,sigmamin

# Define tuples (algorithm setup function, algorithm function, name)
gap = (gap_paramsetup, pyCSalgos.GAP.GAP.GAP, 'GAP')
sl0 = (sl0_paramsetup, pyCSalgos.SL0.SL0_approx.SL0_approx, 'SL0_approx')

# Main function
def mainrun():

  # Define which algorithms to run
  algos = (gap, sl0)
  numalgos = len(algos)
  
  # Set up experiment parameters
  sigma = 2.0;
  delta = 0.8;
  rho   = 0.15;
  numvects = 2; # Number of vectors to generate
  SNRdb = 20;    # This is norm(signal)/norm(noise), so power, not energy

  # Process parameters
  noiselevel = 1 / (10^(SNRdb/10));
  d = 50;
  p = round(sigma*d);
  m = round(delta*d);
  l = round(d - rho*m);
  
  # Generate Omega and data based on parameters
  Omega = pyCSalgos.GAP.GAP.Generate_Analysis_Operator(d, p);
  # Optionally make Omega more coherent
  #[U, S, Vt] = np.linalg.svd(Omega);
  #Sdnew = np.diag(S) * (1+np.arange(np.diag(S).size)); % Make D coherent, not Omega!
  #Snew = [diag(Sdnew); zeros(size(S,1) - size(S,2), size(S,2))];
  #Omega = U * Snew * V';

  # Generate data  
  x0,y,M,Lambda,realnoise = pyCSalgos.GAP.GAP.Generate_Data_Known_Omega(Omega, d,p,m,l,noiselevel, numvects,'l0');

  # Values for lambda
  #lambdas = [0 10.^linspace(-5, 4, 10)];
  lambdas = np.concatenate((np.array([0]), 10**np.linspace(-5, 4, 10)))
  
  xrec = dict()
  err = dict()
  relerr = dict()
  for i,algo in zip(np.arange(numalgos),algos):
    xrec[algo[2]]   = np.zeros((lambdas.size, d, y.shape[1]))
    err[algo[2]]    = np.zeros((lambdas.size, y.shape[1]))
    relerr[algo[2]] = np.zeros((lambdas.size, y.shape[1]))
  
  for ilbd,lbd in zip(np.arange(lambdas.size),lambdas):
    for iy in np.arange(y.shape[1]):
      for algosetupfunc,algofunc,strname in algos:
        epsilon = 1.1 * np.linalg.norm(realnoise[:,iy])
        
        inparams = algosetupfunc(y[:,iy],M,Omega,epsilon,lbd)
        xrec[strname][ilbd,:,iy] = algofunc(*inparams)[0]
        
        err[strname][ilbd,iy]    = np.linalg.norm(x0[:,iy] - xrec[strname][ilbd,:,iy])
        relerr[strname][ilbd,iy] = err[strname][ilbd,iy] / np.linalg.norm(x0[:,iy])
        
    print 'Lambda = ',lbd,' :'
    for strname in relerr:
      print '   ',strname,' : avg relative error = ',np.mean(relerr[strname][ilbd,:])



# Script main
if __name__ == "__main__":
  mainrun()