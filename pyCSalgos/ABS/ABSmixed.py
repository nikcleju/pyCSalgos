# -*- coding: utf-8 -*-
"""
Algorithms for approximate analysis recovery based on synthesis solvers (a.k.a. Analysis by Synthesis, ABS).
Approximate reconstruction, ABS-mixed.

Author: Nicolae Cleju
"""
__author__ = "Nicolae Cleju"
__license__ = "GPL"
__email__ = "nikcleju@gmail.com"


import numpy

# Import synthesis solvers from pyCSalgos package
import pyCSalgos.BP.l1qec
import pyCSalgos.SL0.SL0_approx

def bp(y,M,Omega,epsilon, x0, lbtol=1e-3, mu=10, cgtol=1e-8, cgmaxiter=200, verbose=False):
  """
  ABS-mixed: Basis Pursuit (based on l1magic toolbox)
  """  
  N,n = Omega.shape
  D = numpy.linalg.pinv(Omega)
  U,S,Vt = numpy.linalg.svd(D)
  Aeps = numpy.dot(M,D)
  Aexact = Vt[-(N-n):,:]
  
  return numpy.dot(D , pyCSalgos.BP.l1qec.l1qec_logbarrier(x0,Aeps,Aeps.T,y,epsilon,Aexact,Aexact.T,numpy.zeros(N-n), lbtol, mu, cgtol, cgmaxiter, verbose))

def sl0(y,M,Omega,epsilon, sigma_min, sigma_decrease_factor=0.5, mu_0=2, L=3, Aeps_pinv=None, Aexact_pinv=None, true_s=None):
  """
  ABS-mixed: Smooth L0 (SL0)
  """  
  N,n = Omega.shape
  D = numpy.linalg.pinv(Omega)
  U,S,Vt = numpy.linalg.svd(D)
  Aeps = numpy.dot(M,D)
  Aexact = Vt[-(N-n):,:]
  
  #return numpy.dot(D, pyCSalgos.SL0.SL0_approx.SL0_approx_analysis(Aeps,Aexact,y,epsilon,sigma_min,sigma_decrease_factor,mu_0,L,Aeps_pinv,Aexact_pinv,true_s))
  #return numpy.dot(D, pyCSalgos.SL0.SL0_approx.SL0_robust_analysis(Aeps,Aexact,y,epsilon,sigma_min,sigma_decrease_factor,mu_0,L,Aeps_pinv,Aexact_pinv,true_s))
  #return numpy.dot(D, pyCSalgos.SL0.SL0_approx.SL0_approx_analysis_unconstrained(Aeps,Aexact,y,epsilon,sigma_min,sigma_decrease_factor,mu_0,L,Aeps_pinv,Aexact_pinv,true_s))
  return numpy.dot(D, pyCSalgos.SL0.SL0_approx.SL0_approx_analysis_dai(Aeps,Aexact,y,epsilon,sigma_min,sigma_decrease_factor,mu_0,L,Aeps_pinv,Aexact_pinv,true_s))
  