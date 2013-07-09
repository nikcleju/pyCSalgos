# -*- coding: utf-8 -*-
"""
Algorithms for exact analysis recovery based on synthesis solvers (a.k.a. Analysis by Synthesis, ABS).
Exact reconstruction.

Author: Nicolae Cleju
"""
__author__ = "Nicolae Cleju"
__license__ = "GPL"
__email__ = "nikcleju@gmail.com"


import numpy

# Import synthesis solvers from pyCSalgos package
import pyCSalgos.BP.l1eq_pd
import pyCSalgos.BP.cvxopt_lp
import pyCSalgos.OMP.omp_QR
import pyCSalgos.SL0.SL0
import pyCSalgos.TST.RecommendedTST


def bp(y,M,Omega,x0, pdtol=1e-3, pdmaxiter=50, cgtol=1e-8, cgmaxiter=200, verbose=False):
  """
  ABS-exact: Basis Pursuit (based on l1magic toolbox)
  """
  N,n = Omega.shape
  D = numpy.linalg.pinv(Omega)
  U,S,Vt = numpy.linalg.svd(D)
  Aextra = Vt[-(N-n):,:]
  
  # Create aggregate problem
  Atilde = numpy.vstack((numpy.dot(M,D), Aextra))
  ytilde = numpy.concatenate((y,numpy.zeros(N-n)))

  return numpy.dot(D , pyCSalgos.BP.l1eq_pd.l1eq_pd(x0,Atilde,Atilde.T,ytilde, pdtol, pdmaxiter, cgtol, cgmaxiter, verbose))

def bp_cvxopt(y,M,Omega):
  """
  ABS-exact: Basis Pursuit (based cvxopt)
  """
  N,n = Omega.shape
  D = numpy.linalg.pinv(Omega)
  U,S,Vt = numpy.linalg.svd(D)
  Aextra = Vt[-(N-n):,:]
  
  # Create aggregate problem
  Atilde = numpy.vstack((numpy.dot(M,D), Aextra))
  ytilde = numpy.concatenate((y,numpy.zeros(N-n)))

  return numpy.dot(D , pyCSalgos.BP.cvxopt_lp.cvxopt_lp(ytilde, Atilde))


def ompeps(y,M,Omega,epsilon):
  """
  ABS-exact: OMP with stopping criterion residual < epsilon 
  """
  
  N,n = Omega.shape
  D = numpy.linalg.pinv(Omega)
  U,S,Vt = numpy.linalg.svd(D)
  Aextra = Vt[-(N-n):,:]
  
  # Create aggregate problem
  Atilde = numpy.vstack((numpy.dot(M,D), Aextra))
  ytilde = numpy.concatenate((y,numpy.zeros(N-n)))
  
  opts = dict()
  opts['stopCrit'] = 'mse'
  opts['stopTol'] = epsilon
  return numpy.dot(D , pyCSalgos.OMP.omp_QR.greed_omp_qr(ytilde,Atilde,Atilde.shape[1],opts)[0])

def ompk(y,M,Omega,k):
  """
  ABS-exact: OMP with stopping criterion fixed number of atoms = k
  """
  
  N,n = Omega.shape
  D = numpy.linalg.pinv(Omega)
  U,S,Vt = numpy.linalg.svd(D)
  Aextra = Vt[-(N-n):,:]
  
  # Create aggregate problem
  Atilde = numpy.vstack((numpy.dot(M,D), Aextra))
  ytilde = numpy.concatenate((y,numpy.zeros(N-n)))
  
  opts = dict()
  opts['stopTol'] = k
  return numpy.dot(D , pyCSalgos.OMP.omp_QR.greed_omp_qr(ytilde,Atilde,Atilde.shape[1],opts)[0])

def sl0(y,M,Omega, sigma_min, sigma_decrease_factor=0.5, mu_0=2, L=3, true_s=None):
  """
  ABS-exact: Smooth L0 (SL0)
  """
  
  N,n = Omega.shape
  D = numpy.linalg.pinv(Omega)
  U,S,Vt = numpy.linalg.svd(D)
  Aextra = Vt[-(N-n):,:]
  
  # Create aggregate problem
  Atilde = numpy.vstack((numpy.dot(M,D), Aextra))
  ytilde = numpy.concatenate((y,numpy.zeros(N-n)))
  
  return numpy.dot(D, pyCSalgos.SL0.SL0.SL0(Atilde,ytilde,sigma_min,sigma_decrease_factor,mu_0,L,true_s))

def tst_recom(y,M,Omega, nsweep=300, tol=0.00001, xinitial=None, ro=None):
  """
  ABS-exact: Two Stage Thresholding (TST) with optimized parameters (see Maleki & Donoho)
  """
  
  N,n = Omega.shape
  D = numpy.linalg.pinv(Omega)
  U,S,Vt = numpy.linalg.svd(D)
  Aextra = Vt[-(N-n):,:]
  
  # Create aggregate problem
  Atilde = numpy.vstack((numpy.dot(M,D), Aextra))
  ytilde = numpy.concatenate((y,numpy.zeros(N-n)))
  
  return numpy.dot(D, pyCSalgos.TST.RecommendedTST.RecommendedTST(Atilde, ytilde, nsweep, tol, xinitial, ro))
  