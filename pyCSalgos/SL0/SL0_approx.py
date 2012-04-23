# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 21:29:09 2011

@author: Nic
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 18:39:54 2011

@author: Nic
"""

import numpy
import math
#import cvxpy
import EllipseProj



#function s=SL0(A, x, sigma_min, sigma_decrease_factor, mu_0, L, A_pinv, true_s)
def SL0_approx(A, x, eps, sigma_min, sigma_decrease_factor=0.5, mu_0=2, L=3, A_pinv=None, true_s=None):
  
  if A_pinv is None:
    A_pinv = numpy.linalg.pinv(A)
  
  if true_s is not None:
      ShowProgress = True
  else:
      ShowProgress = False
  
  # Initialization
  #s = A\x;
  s = numpy.dot(A_pinv,x)
  sigma = 2.0 * numpy.abs(s).max()
  
  # Main Loop
  while sigma>sigma_min:
      for i in numpy.arange(L):
          delta = OurDelta(s,sigma)
          s = s - mu_0*delta
          # At this point, s no longer exactly satisfies x = A*s
          # The original SL0 algorithm projects s onto {s | x = As} with
          # s = s - numpy.dot(A_pinv,(numpy.dot(A,s)-x))   # Projection
          # We want to project s onto {s | |x-As| < eps}
          # We move onto the direction -A_pinv*(A*s-x), but only with a
          # smaller step:
          direction = numpy.dot(A_pinv,(numpy.dot(A,s)-x))
          if (numpy.linalg.norm(numpy.dot(A,direction)) >= eps):
            s = s - (1.0 - eps/numpy.linalg.norm(numpy.dot(A,direction))) * direction

          #assert(numpy.linalg.norm(x - numpy.dot(A,s)) < eps + 1e-6)          
      
      if ShowProgress:
          #fprintf('     sigma=#f, SNR=#f\n',sigma,estimate_SNR(s,true_s))
          string = '     sigma=%f, SNR=%f\n' % sigma,estimate_SNR(s,true_s)
          print string
      
      sigma = sigma * sigma_decrease_factor
  
  return s


def SL0_approx_cvxpy(A, x, eps, sigma_min, sigma_decrease_factor=0.5, mu_0=2, L=3, A_pinv=None, true_s=None):
  
  if A_pinv is None:
    A_pinv = numpy.linalg.pinv(A)
  
  if true_s is not None:
      ShowProgress = True
  else:
      ShowProgress = False
  
  # Initialization
  #s = A\x;
  s = numpy.dot(A_pinv,x)
  sigma = 2.0 * numpy.abs(s).max()
  
  # Main Loop
  while sigma>sigma_min:
      for i in numpy.arange(L):
          delta = OurDelta(s,sigma)
          s = s - mu_0*delta
          # At this point, s no longer exactly satisfies x = A*s
          # The original SL0 algorithm projects s onto {s | x = As} with
          # s = s - numpy.dot(A_pinv,(numpy.dot(A,s)-x))   # Projection
          # We want to project s onto {s | |x-As| < eps}
          # We move onto the direction -A_pinv*(A*s-x), but only with a
          # smaller step:
          direction = numpy.dot(A_pinv,(numpy.dot(A,s)-x))
          if (numpy.linalg.norm(numpy.dot(A,direction)) >= eps):
            #s = s - (1.0 - eps/numpy.linalg.norm(numpy.dot(A,direction))) * direction
            s = EllipseProj.ellipse_proj_cvxpy(A,x,s,eps)

          #assert(numpy.linalg.norm(x - numpy.dot(A,s)) < eps + 1e-6)          
      
      if ShowProgress:
          #fprintf('     sigma=#f, SNR=#f\n',sigma,estimate_SNR(s,true_s))
          string = '     sigma=%f, SNR=%f\n' % sigma,estimate_SNR(s,true_s)
          print string
      
      sigma = sigma * sigma_decrease_factor
  
  return s


def SL0_approx_dai(A, x, eps, sigma_min, sigma_decrease_factor=0.5, mu_0=2, L=3, A_pinv=None, true_s=None):
  
  if A_pinv is None:
    A_pinv = numpy.linalg.pinv(A)
  
  if true_s is not None:
      ShowProgress = True
  else:
      ShowProgress = False
  
  # Initialization
  #s = A\x;
  s = numpy.dot(A_pinv,x)
  sigma = 2.0 * numpy.abs(s).max()
  
  # Main Loop
  while sigma>sigma_min:
      for i in numpy.arange(L):
          delta = OurDelta(s,sigma)
          s = s - mu_0*delta
          # At this point, s no longer exactly satisfies x = A*s
          # The original SL0 algorithm projects s onto {s | x = As} with
          # s = s - numpy.dot(A_pinv,(numpy.dot(A,s)-x))   # Projection
          # We want to project s onto {s | |x-As| < eps}
          # We move onto the direction -A_pinv*(A*s-x), but only with a
          # smaller step:
          direction = numpy.dot(A_pinv,(numpy.dot(A,s)-x))
          if (numpy.linalg.norm(numpy.dot(A,direction)) >= eps):
            #s = s - (1.0 - eps/numpy.linalg.norm(numpy.dot(A,direction))) * direction
            try:
              s = EllipseProj.ellipse_proj_dai(A,x,s,eps)
            except Exception, e:
              #raise EllipseProj.EllipseProjDaiError(e)
              raise EllipseProj.EllipseProjDaiError()
              

          #assert(numpy.linalg.norm(x - numpy.dot(A,s)) < eps + 1e-6)          
      
      if ShowProgress:
          #fprintf('     sigma=#f, SNR=#f\n',sigma,estimate_SNR(s,true_s))
          string = '     sigma=%f, SNR=%f\n' % sigma,estimate_SNR(s,true_s)
          print string
      
      sigma = sigma * sigma_decrease_factor
  
  return s

def SL0_approx_proj(A, x, eps, sigma_min, sigma_decrease_factor=0.5, mu_0=2, L=3, L2=3, A_pinv=None, true_s=None):
  
  if A_pinv is None:
    A_pinv = numpy.linalg.pinv(A)
  
  if true_s is not None:
      ShowProgress = True
  else:
      ShowProgress = False
  
  # Initialization
  #s = A\x;
  s = numpy.dot(A_pinv,x)
  sigma = 2.0 * numpy.abs(s).max()
  
  u,singvals,v  = numpy.linalg.svd(A, full_matrices=0)
  
  # Main Loop
  while sigma>sigma_min:
      for i in numpy.arange(L):
          delta = OurDelta(s,sigma)
          s = s - mu_0*delta
          # At this point, s no longer exactly satisfies x = A*s
          # The original SL0 algorithm projects s onto {s | x = As} with
          # s = s - numpy.dot(A_pinv,(numpy.dot(A,s)-x))   # Projection
          # We want to project s onto {s | |x-As| < eps}
          # We move onto the direction -A_pinv*(A*s-x), but only with a
          # smaller step:
          s_orig = s

          # Reference
          direction = numpy.dot(A_pinv,(numpy.dot(A,s)-x))
          if (numpy.linalg.norm(numpy.dot(A,direction)) >= eps):
            #s = s - (1.0 - eps/numpy.linalg.norm(numpy.dot(A,direction))) * direction
            s_cvxpy = EllipseProj.ellipse_proj_cvxpy(A,x,s,eps)       
            
          # Starting point
          direction = numpy.dot(A_pinv,(numpy.dot(A,s)-x))
          if (numpy.linalg.norm(numpy.dot(A,direction)) >= eps):
            s_first = s - (1.0 - eps/numpy.linalg.norm(numpy.dot(A,direction))) * direction

          #steps = 1
          ##steps = math.floor(math.log2(numpy.lingl.norm(s)/eps))
          #step = math.pow(numpy.linalg.norm(s)/eps, 1.0/steps)
          #eps = eps * step**(steps-1)
          #for k in range(steps):
            
          direction = numpy.dot(A_pinv,(numpy.dot(A,s)-x))
          if (numpy.linalg.norm(numpy.dot(A,direction)) >= eps):
            s = EllipseProj.ellipse_proj_proj(A,x,s,eps,L2)
            
            #eps = eps/step
      
      if ShowProgress:
          #fprintf('     sigma=#f, SNR=#f\n',sigma,estimate_SNR(s,true_s))
          string = '     sigma=%f, SNR=%f\n' % sigma,estimate_SNR(s,true_s)
          print string
      
      sigma = sigma * sigma_decrease_factor
  
  return s
  
def SL0_approx_unconstrained(A, x, eps, lmbda, sigma_min, sigma_decrease_factor=0.5, mu_0=2, L=3, A_pinv=None, true_s=None):
  
  if A_pinv is None:
    A_pinv = numpy.linalg.pinv(A)
  
  if true_s is not None:
      ShowProgress = True
  else:
      ShowProgress = False
  
  # Initialization
  #s = A\x;
  s = numpy.dot(A_pinv,x)
  sigma = 2.0 * numpy.abs(s).max()
  
  lmbda =   1./(0.007 + 3.5*(eps/x.size)**2)*1e-5
  #lmbda = 0.5
  mu = mu_0  
  
  # Main Loop
  while sigma>sigma_min:
      for i in numpy.arange(L):
          #delta = OurDeltaUnconstrained(s,sigma,lmbda,A,x)
          delta = s * numpy.exp( (-numpy.abs(s)**2) / sigma**2) + (sigma**2)*lmbda*2*numpy.dot(A.T, numpy.dot(A,s)-x)
          snew = s - mu*delta
          
          Js    =    s.size - numpy.sum(numpy.exp( (-numpy.abs(s)**2)    / sigma**2)) + lmbda*numpy.linalg.norm(numpy.dot(A,s)   -x)**2
          Jsnew = snew.size - numpy.sum(numpy.exp( (-numpy.abs(snew)**2) / sigma**2)) + lmbda*numpy.linalg.norm(numpy.dot(A,snew)-x)**2
          
          #if Jsnew < Js:
          #  rho = 1.2
          #else:
          #  rho = 0.5
          rho = 1

          #s = s - mu*delta
          s = snew.copy()
          
          mu = mu * rho
      
      if ShowProgress:
          #fprintf('     sigma=#f, SNR=#f\n',sigma,estimate_SNR(s,true_s))
          string = '     sigma=%f, SNR=%f\n' % sigma,estimate_SNR(s,true_s)
          print string
      
      sigma = sigma * sigma_decrease_factor
  
  return s

# Direct approximate analysis-based version of SL0
# Solves argimn_gamma ||gamma||_0 such that ||x - Aeps*gamma|| < eps AND ||Aexact*gamma|| = 0
# Basically instead of having a single A, we now have one Aeps which is up to eps error
#  and an Axeact which requires exact orthogonality
# It is assumed that the rows of Aexact are orthogonal to the rows of Aeps
def SL0_approx_analysis(Aeps, Aexact, x, eps, sigma_min, sigma_decrease_factor=0.5, mu_0=2, L=3, Aeps_pinv=None, Aexact_pinv=None, true_s=None):
  
  if Aeps_pinv is None:
    Aeps_pinv = numpy.linalg.pinv(Aeps)
  if Aexact_pinv is None:
    Aexact_pinv = numpy.linalg.pinv(Aexact)
    
  if true_s is not None:
      ShowProgress = True
  else:
      ShowProgress = False
  
  # Initialization
  #s = A\x;
  s = numpy.dot(Aeps_pinv,x)
  sigma = 2.0 * numpy.abs(s).max()
  
  # Main Loop
  while sigma>sigma_min:
      for i in numpy.arange(L):
          delta = OurDelta(s,sigma)
          s = s - mu_0*delta
          # At this point, s no longer exactly satisfies x = A*s
          # The original SL0 algorithm projects s onto {s | x = As} with
          # s = s - numpy.dot(A_pinv,(numpy.dot(A,s)-x))   # Projection
          #
          # We want to project s onto {s | |x-AEPS*s|<eps AND |Aexact*s|=0}
          # First:   make s orthogonal to Aexact (|Aexact*s|=0)
          # Second:  move onto the direction -A_pinv*(A*s-x), but only with a smaller step:
          # This separation assumes that the rows of Aexact are orthogonal to the rows of Aeps
          #
          # 1. Make s orthogonal to Aexact:
          #     s = s - Aexact_pinv * Aexact * s
          s = s - numpy.dot(Aexact_pinv,(numpy.dot(Aexact,s)))
          # 2. Move onto the direction -A_pinv*(A*s-x), but only with a smaller step:
          direction = numpy.dot(Aeps_pinv,(numpy.dot(Aeps,s)-x))
          # Nic 10.04.2012: Why numpy.dot(Aeps,direction) and not just 'direction'?
          # Nic 10.04.2012: because 'direction' is of size(s), but I'm interested in it's projection on Aeps
          if (numpy.linalg.norm(numpy.dot(Aeps,direction)) >= eps):
            s = s - (1.0 - eps/numpy.linalg.norm(numpy.dot(Aeps,direction))) * direction

          #assert(numpy.linalg.norm(x - numpy.dot(A,s)) < eps + 1e-6)          
      
      if ShowProgress:
          #fprintf('     sigma=#f, SNR=#f\n',sigma,estimate_SNR(s,true_s))
          string = '     sigma=%f, SNR=%f\n' % sigma,estimate_SNR(s,true_s)
          print string
      
      sigma = sigma * sigma_decrease_factor
  
  return s
  

def SL0_approx_analysis_cvxpy(Aeps, Aexact, x, eps, sigma_min, sigma_decrease_factor=0.5, mu_0=2, L=3, Aeps_pinv=None, Aexact_pinv=None, true_s=None):
  
  if Aeps_pinv is None:
    Aeps_pinv = numpy.linalg.pinv(Aeps)
  if Aexact_pinv is None:
    Aexact_pinv = numpy.linalg.pinv(Aexact)
    
  if true_s is not None:
      ShowProgress = True
  else:
      ShowProgress = False
  
  # Initialization
  #s = A\x;
  s = numpy.dot(Aeps_pinv,x)
  sigma = 2.0 * numpy.abs(s).max()
  
  # Main Loop
  while sigma>sigma_min:
      for i in numpy.arange(L):
          delta = OurDelta(s,sigma)
          s = s - mu_0*delta
          # At this point, s no longer exactly satisfies x = A*s
          # The original SL0 algorithm projects s onto {s | x = As} with
          # s = s - numpy.dot(A_pinv,(numpy.dot(A,s)-x))   # Projection
          #
          # We want to project s onto {s | |x-AEPS*s|<eps AND |Aexact*s|=0}
          # First:   make s orthogonal to Aexact (|Aexact*s|=0)
          # Second:  move onto the direction -A_pinv*(A*s-x), but only with a smaller step:
          # This separation assumes that the rows of Aexact are orthogonal to the rows of Aeps
          #
          # 1. Make s orthogonal to Aexact:
          #     s = s - Aexact_pinv * Aexact * s
          s = s - numpy.dot(Aexact_pinv,(numpy.dot(Aexact,s)))
          # 2. Move onto the direction -A_pinv*(A*s-x), but only with a smaller step:
          direction = numpy.dot(Aeps_pinv,(numpy.dot(Aeps,s)-x))
          # Nic 10.04.2012: Why numpy.dot(Aeps,direction) and not just 'direction'?
          # Nic 10.04.2012: because 'direction' is of size(s), but I'm interested in it's projection on Aeps
          if (numpy.linalg.norm(numpy.dot(Aeps,direction)) >= eps):
            #  s = s - (1.0 - eps/numpy.linalg.norm(numpy.dot(Aeps,direction))) * direction
            s = EllipseProj.ellipse_proj_cvxpy(Aeps,x,s,eps)
            #s = EllipseProj.ellipse_proj_logbarrier(Aeps,x,s,eps)

          #assert(numpy.linalg.norm(x - numpy.dot(A,s)) < eps + 1e-6)          
      
      if ShowProgress:
          #fprintf('     sigma=#f, SNR=#f\n',sigma,estimate_SNR(s,true_s))
          string = '     sigma=%f, SNR=%f\n' % sigma,estimate_SNR(s,true_s)
          print string
      
      sigma = sigma * sigma_decrease_factor
  
  return s


def SL0_approx_analysis_dai(Aeps, Aexact, x, eps, sigma_min, sigma_decrease_factor=0.5, mu_0=2, L=3, Aeps_pinv=None, Aexact_pinv=None, true_s=None):
  
  if Aeps_pinv is None:
    Aeps_pinv = numpy.linalg.pinv(Aeps)
  if Aexact_pinv is None:
    Aexact_pinv = numpy.linalg.pinv(Aexact)
    
  if true_s is not None:
      ShowProgress = True
  else:
      ShowProgress = False
  
  # Initialization
  #s = A\x;
  s = numpy.dot(Aeps_pinv,x)
  sigma = 2.0 * numpy.abs(s).max()
  
  # Main Loop
  while sigma>sigma_min:
      for i in numpy.arange(L):
          delta = OurDelta(s,sigma)
          s = s - mu_0*delta
          # At this point, s no longer exactly satisfies x = A*s
          # The original SL0 algorithm projects s onto {s | x = As} with
          # s = s - numpy.dot(A_pinv,(numpy.dot(A,s)-x))   # Projection
          #
          # We want to project s onto {s | |x-AEPS*s|<eps AND |Aexact*s|=0}
          # First:   make s orthogonal to Aexact (|Aexact*s|=0)
          # Second:  move onto the direction -A_pinv*(A*s-x), but only with a smaller step:
          # This separation assumes that the rows of Aexact are orthogonal to the rows of Aeps
          #
          # 1. Make s orthogonal to Aexact:
          #     s = s - Aexact_pinv * Aexact * s
          s = s - numpy.dot(Aexact_pinv,(numpy.dot(Aexact,s)))
          # 2. Move onto the direction -A_pinv*(A*s-x), but only with a smaller step:
          direction = numpy.dot(Aeps_pinv,(numpy.dot(Aeps,s)-x))
          # Nic 10.04.2012: Why numpy.dot(Aeps,direction) and not just 'direction'?
          # Nic 10.04.2012: because 'direction' is of size(s), but I'm interested in it's projection on Aeps
          if (numpy.linalg.norm(numpy.dot(Aeps,direction)) >= eps):
            #  s = s - (1.0 - eps/numpy.linalg.norm(numpy.dot(Aeps,direction))) * direction
            try:
              s = EllipseProj.ellipse_proj_dai(Aeps,x,s,eps)
            except Exception, e:
              #raise EllipseProj.EllipseProjDaiError(e)
              raise EllipseProj.EllipseProjDaiError()

          #assert(numpy.linalg.norm(x - numpy.dot(A,s)) < eps + 1e-6)          
      
      if ShowProgress:
          #fprintf('     sigma=#f, SNR=#f\n',sigma,estimate_SNR(s,true_s))
          string = '     sigma=%f, SNR=%f\n' % sigma,estimate_SNR(s,true_s)
          print string
      
      sigma = sigma * sigma_decrease_factor
  
  return s

def SL0_robust_analysis(Aeps, Aexact, x, eps, sigma_min, sigma_decrease_factor=0.5, mu_0=2, L=3, Aeps_pinv=None, Aexact_pinv=None, true_s=None):
  
  if Aeps_pinv is None:
    Aeps_pinv = numpy.linalg.pinv(Aeps)
  if Aexact_pinv is None:
    Aexact_pinv = numpy.linalg.pinv(Aexact)
    
  if true_s is not None:
      ShowProgress = True
  else:
      ShowProgress = False
  
  # Initialization
  #s = A\x;
  s = numpy.dot(Aeps_pinv,x)
  sigma = 2.0 * numpy.abs(s).max()
  
  # Main Loop
  while sigma>sigma_min:
      for i in numpy.arange(L):
          delta = OurDelta(s,sigma)
          s = s - mu_0*delta
          # At this point, s no longer exactly satisfies x = A*s
          # The original SL0 algorithm projects s onto {s | x = As} with
          # s = s - numpy.dot(A_pinv,(numpy.dot(A,s)-x))   # Projection
          #
          # 1. Make s orthogonal to Aexact:
          #     s = s - Aexact_pinv * Aexact * s
          s = s - numpy.dot(Aexact_pinv,(numpy.dot(Aexact,s)))
          # 2. 
          if (numpy.linalg.norm(numpy.dot(Aeps,s) - x) >= eps):
            s = s - numpy.dot(Aeps.T , numpy.dot(numpy.linalg.inv(numpy.dot(Aeps,Aeps.T)), numpy.dot(Aeps,s)-x))
      
      if ShowProgress:
          #fprintf('     sigma=#f, SNR=#f\n',sigma,estimate_SNR(s,true_s))
          string = '     sigma=%f, SNR=%f\n' % sigma,estimate_SNR(s,true_s)
          print string
      
      sigma = sigma * sigma_decrease_factor
  
  return s
 
#def SL0_approx_analysis_unconstrained(Aeps, Aexact, x, eps, sigma_min, sigma_decrease_factor=0.5, mu_0=2, L=3, Aeps_pinv=None, Aexact_pinv=None, true_s=None):
#  
#  if Aeps_pinv is None:
#    Aeps_pinv = numpy.linalg.pinv(Aeps)
#  if Aexact_pinv is None:
#    Aexact_pinv = numpy.linalg.pinv(Aexact)
#    
#  if true_s is not None:
#      ShowProgress = True
#  else:
#      ShowProgress = False
#  
#  # Initialization
#  #s = A\x;
#  s = numpy.dot(Aeps_pinv,x)
#  sigma = 2.0 * numpy.abs(s).max()
#  
#  lmbda_orig = 1./(0.007 + 3.5*(eps/x.size)**2)
#  #lmbda = 100/sigma**2
#  mu = mu_0
#  
#  # Main Loop
#  while sigma>sigma_min:
#    
#      lmbda = lmbda_orig * sigma**2
#      
#      for i in numpy.arange(L):
#          delta = OurDeltaUnconstrained(s,sigma,lmbda,Aeps,x)
#          snew = s - mu*delta
#          
#          Js    =    s.size - numpy.sum(numpy.exp( (-numpy.abs(s)**2)    / sigma**2)) + lmbda*numpy.linalg.norm(numpy.dot(Aeps,s)   -x)**2
#          Jsnew = snew.size - numpy.sum(numpy.exp( (-numpy.abs(snew)**2) / sigma**2)) + lmbda*numpy.linalg.norm(numpy.dot(Aeps,snew)-x)**2
#          
#          if Jsnew < Js:
#            rho = 1.2
#          else:
#            rho = 0.5
#
#          #s = s - mu*delta
#          s = snew.copy()
#          
#          mu = mu * rho
#          
#          
#          # At this point, s no longer exactly satisfies x = A*s
#          # The original SL0 algorithm projects s onto {s | x = As} with
#          # s = s - numpy.dot(A_pinv,(numpy.dot(A,s)-x))   # Projection
#          #
#          # We want to project s onto {s | |x-AEPS*s|<eps AND |Aexact*s|=0}
#          # First:   make s orthogonal to Aexact (|Aexact*s|=0)
#          # Second:  move onto the direction -A_pinv*(A*s-x), but only with a smaller step:
#          # This separation assumes that the rows of Aexact are orthogonal to the rows of Aeps
#          #
#          # 1. Make s orthogonal to Aexact:
#          #     s = s - Aexact_pinv * Aexact * s
#          s = s - numpy.dot(Aexact_pinv,(numpy.dot(Aexact,s)))
#          # 2. 
#
#          #assert(numpy.linalg.norm(x - numpy.dot(A,s)) < eps + 1e-6)          
#      
#      if ShowProgress:
#          #fprintf('     sigma=#f, SNR=#f\n',sigma,estimate_SNR(s,true_s))
#          string = '     sigma=%f, SNR=%f\n' % sigma,estimate_SNR(s,true_s)
#          print string
#      
#      sigma = sigma * sigma_decrease_factor
#      lmbda = 100/sigma**2
#  
#  return s 
 
####################################################################
#function delta=OurDelta(s,sigma)
def OurDelta(s,sigma):
  
  return s * numpy.exp( (-numpy.abs(s)**2) / sigma**2)


def OurDeltaUnconstrained(s,sigma,lmbda,A,x):
  
  return s * numpy.exp( (-numpy.abs(s)**2) / sigma**2) + lmbda*2*numpy.dot(A.T, numpy.dot(A,s)-x)

  
####################################################################
#function SNR=estimate_SNR(estim_s,true_s)
def estimate_SNR(estim_s, true_s):
  
  err = true_s - estim_s
  return 10*numpy.log10((true_s**2).sum()/(err**2).sum())


