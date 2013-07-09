# -*- coding: utf-8 -*-
"""
Algorithms for approximate analysis recovery based on synthesis solvers (a.k.a. Analysis by Synthesis, ABS).
Approximate reconstruction, ABS-lambda.

Author: Nicolae Cleju
"""
__author__ = "Nicolae Cleju"
__license__ = "GPL"
__email__ = "nikcleju@gmail.com"


import numpy

# Import synthesis solvers from pyCSalgos package
import pyCSalgos.BP.l1qc
import pyCSalgos.SL0.SL0_approx
import pyCSalgos.OMP.omp_QR
import pyCSalgos.TST.RecommendedTST

def sl0(y,M,Omega,epsilon,lbd,sigma_min, sigma_decrease_factor=0.5, mu_0=2, L=3, A_pinv=None, true_s=None):
  """
  ABS-lambda: Smooth L0 (SL0)
  """    
  N,n = Omega.shape
  D = numpy.linalg.pinv(Omega)
  U,S,Vt = numpy.linalg.svd(D)
  aggDupper = numpy.dot(M,D)
  aggDlower = Vt[-(N-n):,:]
  aggD = numpy.vstack((aggDupper, lbd * aggDlower))
  aggy = numpy.concatenate((y, numpy.zeros(N-n)))
  
  #return numpy.dot(D, pyCSalgos.SL0.SL0_approx.SL0_approx(aggD,aggy,epsilon,sigma_min,sigma_decrease_factor,mu_0,L,A_pinv,true_s))
  return numpy.dot(D, pyCSalgos.SL0.SL0_approx.SL0_approx_dai(aggD,aggy,epsilon,sigma_min,sigma_decrease_factor,mu_0,L,A_pinv,true_s))
  
def bp(y,M,Omega,epsilon,lbd, x0, lbtol=1e-3, mu=10, cgtol=1e-8, cgmaxiter=200, verbose=False):
  """
  ABS-lambda: Basis Pursuit (based on l1magic toolbox)
  """    
  N,n = Omega.shape
  D = numpy.linalg.pinv(Omega)
  U,S,Vt = numpy.linalg.svd(D)
  aggDupper = numpy.dot(M,D)
  aggDlower = Vt[-(N-n):,:]
  aggD = numpy.vstack((aggDupper, lbd * aggDlower))
  aggy = numpy.concatenate((y, numpy.zeros(N-n)))

  return numpy.dot(D, pyCSalgos.BP.l1qc.l1qc_logbarrier(x0,aggD,aggD.T,aggy,epsilon, lbtol, mu, cgtol, cgmaxiter, verbose))

def ompeps(y,M,Omega,epsilon,lbd):
  """
  ABS-lambda: OMP with stopping criterion residual < epsilon 
  """
  N,n = Omega.shape
  D = numpy.linalg.pinv(Omega)
  U,S,Vt = numpy.linalg.svd(D)
  aggDupper = numpy.dot(M,D)
  aggDlower = Vt[-(N-n):,:]
  aggD = numpy.vstack((aggDupper, lbd * aggDlower))
  aggy = numpy.concatenate((y, numpy.zeros(N-n)))
  
  opts = dict()
  opts['stopCrit'] = 'mse'
  opts['stopTol'] = epsilon**2 / aggy.size
  return numpy.dot(D, pyCSalgos.OMP.omp_QR.greed_omp_qr(aggy,aggD,aggD.shape[1],opts)[0])
  
def tst_recom(y,M,Omega,epsilon,lbd, nsweep=300, xinitial=None, ro=None):
  """
  ABS-lambda: Two Stage Thresholding (TST) with optimized parameters (see Maleki & Donoho)
  """  
  N,n = Omega.shape
  D = numpy.linalg.pinv(Omega)
  U,S,Vt = numpy.linalg.svd(D)
  aggDupper = numpy.dot(M,D)
  aggDlower = Vt[-(N-n):,:]
  aggD = numpy.vstack((aggDupper, lbd * aggDlower))
  aggy = numpy.concatenate((y, numpy.zeros(N-n)))
  
  tol = epsilon / numpy.linalg.norm(aggy)
  return numpy.dot(D, pyCSalgos.TST.RecommendedTST.RecommendedTST(aggD, aggy, nsweep, tol, xinitial, ro))
  