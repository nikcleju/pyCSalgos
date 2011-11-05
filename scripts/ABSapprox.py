# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 18:08:40 2011

@author: Nic
"""

import numpy
import pyCSalgos

def gap_paramsetup(y,M,Omega,epsilon,lbd):
  gapparams = dict(num_iteration = 1000,
                   greedy_level = 0.9,
                   stopping_coefficientstopping_coefficient_size = 1e-4,
                   l2solver = 'pseudoinverse',
                   noise_level = epsilon)
  return y,M,M.T,Omega,Omega.T,gapparams,numpy.zeros(Omega.shape[1])

def omp_paramsetup(y,M,Omega,epsilon,lbd):
  gapparams = dict(num_iteration = 1000,
                   greedy_level = 0.9,
                   stopping_coefficientstopping_coefficient_size = 1e-4,
                   l2solver = 'pseudoinverse',
                   noise_level = epsilon)
  return y,M,M.T,Omega,Omega.T,gapparams,numpy.zeros(Omega.shape[1])

gap = (pyCSalgos.GAP, gap_paramsetup)



gap = (pyCSalgos.GAP, gap_paramsetup)
  



def mainrun():
  
  algos = (gap, sl0)
  
  for algofunc,paramsetup in algos:
    xrec = algofunc(algosetup(y, Omega, epsilon, lbd))
