# -*- coding: utf-8 -*-
"""
Define simple wrappers for algorithms, with similar header.
Specific algorithm parameters are defined inside here.

Author: Nicolae Cleju
"""
__author__ = "Nicolae Cleju"
__license__ = "GPL"
__email__ = "nikcleju@gmail.com"


import numpy

# Module with algorithms implemented in Python
import pyCSalgos
import pyCSalgos.GAP.GAP

# Analysis by Synthesis - exact algorithms
import ABSexact
# Analysis by Synthesis - mixed algorithms
import ABSmixed
# Analysis by Synthesis - lambda algorithms
import ABSlambda


###---------------------------------
### Exact reconstruction algorithms
###---------------------------------
def run_exact_gap(y,M,Omega):
  """
  Wrapper for GAP algorithm for exact analysis recovery
  """
  gapparams = {"num_iteration" : 1000,\
                   "greedy_level" : 0.9,\
                   "stopping_coefficient_size" : 1e-4,\
                   "l2solver" : 'pseudoinverse',\
                   "noise_level": 1e-10}
  return pyCSalgos.GAP.GAP.GAP(y,M,M.T,Omega,Omega.T,gapparams,numpy.zeros(Omega.shape[1]))[0]
    
def run_exact_bp(y,M,Omega):
  """
  Wrapper for BP algorithm for exact analysis recovery
  Algorithm implementation is l1eq_pd() from l1-magic toolbox
  """
  return ABSexact.bp(y,M,Omega,numpy.zeros(Omega.shape[0]), pdtol=1e-5, pdmaxiter = 100)

def run_exact_bp_cvxopt(y,M,Omega):
  """
  Wrapper for BP algorithm for exact analysis recovery
  Algorithm implementation is using cvxopt linear programming
  """
  return ABSexact.bp_cvxopt(y,M,Omega)


def run_exact_ompeps(y,M,Omega):
  """
  Wrapper for OMP algorithm for exact analysis recovery, with stopping criterion = epsilon
  """
  return ABSexact.ompeps(y,M,Omega,1e-9)
  
#def run_exact_ompk(y,M,Omega)
#  """
#  Wrapper for OMP algorithm for exact analysis recovery, with stopping criterion = fixed no. of atoms
#  """

def run_exact_sl0(y,M,Omega):
  """
  Wrapper for SL0 algorithm for exact analysis recovery
  """
  sigma_min = 1e-12
  sigma_decrease_factor = 0.5
  mu_0 = 2
  L = 20
  return ABSexact.sl0(y,M,Omega, sigma_min, sigma_decrease_factor, mu_0, L)

def run_exact_tst(y,M,Omega):
  """
  Wrapper for TST algorithm (with default optimized params) for exact analysis recovery
  """
  nsweep = 300
  tol = 1e-5
  return ABSexact.tst_recom(y,M,Omega, nsweep, tol)


###---------------------------------------
### Approximate reconstruction algorithms
###---------------------------------------
#  1. Native

def run_gap(y,M,Omega,epsilon):
  """
  Wrapper for GAP algorithm for approximate analysis recovery
  """
  gapparams = {"num_iteration" : 1000,\
                   "greedy_level" : 0.9,\
                   "stopping_coefficient_size" : 1e-4,\
                   "l2solver" : 'pseudoinverse',\
                   "noise_level": epsilon}
  return pyCSalgos.GAP.GAP.GAP(y,M,M.T,Omega,Omega.T,gapparams,numpy.zeros(Omega.shape[1]))[0]

def run_nesta(y,M,Omega,epsilon):
  """
  Wrapper for NESTA algorithm for approximate analysis recovery
  """  
  U,S,V = numpy.linalg.svd(M, full_matrices = True)
  V = V.T         # Make like Matlab
  m,n = M.shape   # Make like Matlab
  S = numpy.hstack((numpy.diag(S), numpy.zeros((m,n-m))))  

  opt_muf = 1e-3
  optsUSV = {'U':U, 'S':S, 'V':V}
  opts = {'U':Omega, 'Ut':Omega.T.copy(), 'USV':optsUSV, 'TolVar':1e-5, 'Verbose':0}
  return pyCSalgos.NESTA.NESTA.NESTA(M, None, y, opt_muf, epsilon, opts)[0]

#  2. ABS-mixed

def run_mixed_sl0(y,M,Omega,epsilon):
  """
  Wrapper for SL0-mixed algorithm for approximate analysis recovery
  """ 
  sigma_min = 0.001
  sigma_decrease_factor = 0.5
  mu_0 = 2
  L = 10
  return ABSmixed.sl0(y,M,Omega,epsilon,sigma_min, sigma_decrease_factor, mu_0, L)

def run_mixed_bp(y,M,Omega,epsilon):
  """
  Wrapper for BP-mixed algorithm for approximate analysis recovery
  """   
  return ABSmixed.bp(y,M,Omega,epsilon, numpy.zeros(Omega.shape[0]))

#  3. ABS-lambda

def run_lambda_sl0(y,M,Omega,epsilon,lbd):
  """
  Wrapper for SL0 algorithm within ABS-lambda approach for approximate analysis recovery
  """    
  sigma_min = 0.001
  sigma_decrease_factor = 0.5
  mu_0 = 2
  L = 10
  return ABSlambda.sl0(y,M,Omega,epsilon, lbd, sigma_min, sigma_decrease_factor, mu_0, L)

def run_lambda_bp(y,M,Omega,epsilon,lbd):
  """
  Wrapper for BP algorithm within ABS-lambda approach for approximate analysis recovery
  """     
  return ABSlambda.bp(y,M,Omega,epsilon,lbd,numpy.zeros(Omega.shape[0]))

def run_lambda_ompeps(y,M,Omega,epsilon,lbd):
  """
  Wrapper for OMP algorithm, with stopping criterion = epsilon,
  for approximate analysis recovery within ABS-lambda approach
  """     
  return ABSlambda.ompeps(y,M,Omega,epsilon,lbd)

def run_lambda_tst(y,M,Omega,epsilon,lbd):
  """
  Wrapper for TST algorithm (with default optimized params)
  for approximate analysis recovery within ABS-lambda approach
  """
  nsweep = 300
  return ABSlambda.tst_recom(y,M,Omega,epsilon,lbd, nsweep)
  
  
### Define algorithm tuples: (function, name)
### Will be used in stdparams and in test scripts
## Exact recovery
exact_gap = (run_exact_gap, 'GAP')
exact_bp = (run_exact_bp, 'ABSexact_BP')
exact_bp_cvxopt = (run_exact_bp_cvxopt, 'ABSexact_BP_cvxopt')
exact_ompeps = (run_exact_ompeps, 'ABSexact_OMPeps')
exact_sl0 = (run_exact_sl0, 'ABSexact_SL0')
exact_tst = (run_exact_tst, 'ABSexact_TST')
## Approximate recovery
# Native
gap = (run_gap, 'GAP')
nesta = (run_nesta, 'NESTA')
# ABS-mixed
mixed_sl0 = (run_mixed_sl0, 'ABSmixed_SL0')
mixed_bp = (run_mixed_bp, 'ABSmixed_BP')
# ABS-lambda
lambda_sl0 = (run_lambda_sl0, 'ABSlambda_SL0')
lambda_bp = (run_lambda_bp, 'ABSlambda_BP')
lambda_ompeps = (run_lambda_ompeps, 'ABSlambda_OMPeps')
lambda_tst = (run_lambda_tst, 'ABSlambda_TST')
