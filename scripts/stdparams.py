# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 14:04:40 2011

@author: ncleju
"""

import numpy
from algos import *

#==========================
# Standard parameters
#==========================
# Standard parameters for quick testing
# Algorithms: GAP, SL0 and BP
# d=50, sigma = 2, delta and rho only 3 x 3, lambdas = 0, 1e-4, 1e-2, 1, 100, 10000
# Do save data, do save plots, don't show plots
# Useful for short testing 
def stdtest():
  # Define which algorithms to run
  algosN = nesta,      # tuple of algorithms not depending on lambda
  #algosL = sl0,bp    # tuple of algorithms depending on lambda (our ABS approach)
  algosL = ()
  
  d = 50.0
  sigma = 2.0
  deltas = numpy.array([0.05, 0.45, 0.95])
  rhos = numpy.array([0.05, 0.45, 0.95])
  #deltas = numpy.array([0.95])
  #deltas = numpy.arange(0.05,1.,0.05)
  #rhos = numpy.array([0.05])
  numvects = 10; # Number of vectors to generate
  SNRdb = 20.;    # This is norm(signal)/norm(noise), so power, not energy
  # Values for lambda
  #lambdas = [0 10.^linspace(-5, 4, 10)];
  lambdas = numpy.array([0., 0.0001, 0.01, 1, 100, 10000])
  
  dosavedata = True
  savedataname = 'approx_pt_stdtest.mat'
  doshowplot = False
  dosaveplot = True
  saveplotbase = 'approx_pt_stdtest_'
  saveplotexts = ('png','pdf','eps')

  return algosN,algosL,d,sigma,deltas,rhos,lambdas,numvects,SNRdb,dosavedata,savedataname,\
          doshowplot,dosaveplot,saveplotbase,saveplotexts   


# Standard parameters 1
# All algorithms, 100 vectors
# d=50, sigma = 2, delta and rho full resolution (0.05 step), lambdas = 0, 1e-4, 1e-2, 1, 100, 10000
# Do save data, do save plots, don't show plots
def std1():
  # Define which algorithms to run
  algosN = gap,sl0analysis,bpanalysis,nesta       # tuple of algorithms not depending on lambda
  algosL = sl0,bp,ompeps,tst    # tuple of algorithms depending on lambda (our ABS approach)
  
  d = 50.0;
  sigma = 2.0
  deltas = numpy.arange(0.05,1.,0.05)
  rhos = numpy.arange(0.05,1.,0.05)
  numvects = 100; # Number of vectors to generate
  SNRdb = 20.;    # This is norm(signal)/norm(noise), so power, not energy
  # Values for lambda
  #lambdas = [0 10.^linspace(-5, 4, 10)];
  lambdas = numpy.array([0., 0.0001, 0.01, 1, 100, 10000])
  
  dosavedata = True
  savedataname = 'approx_pt_std1.mat'
  doshowplot = False
  dosaveplot = True
  saveplotbase = 'approx_pt_std1_'
  saveplotexts = ('png','pdf','eps')

  return algosN,algosL,d,sigma,deltas,rhos,lambdas,numvects,SNRdb,dosavedata,savedataname,\
          doshowplot,dosaveplot,saveplotbase,saveplotexts
          
         
# Standard parameters 2
# All algorithms, 100 vectors
# d=20, sigma = 10, delta and rho full resolution (0.05 step), lambdas = 0, 1e-4, 1e-2, 1, 100, 10000
# Do save data, do save plots, don't show plots
def std2():
  # Define which algorithms to run
  algosN = gap,sl0analysis,bpanalysis,nesta      # tuple of algorithms not depending on lambda
  algosL = sl0,bp,ompeps,tst    # tuple of algorithms depending on lambda (our ABS approach)
  
  d = 20.0
  sigma = 10.0
  deltas = numpy.arange(0.05,1.,0.05)
  rhos = numpy.arange(0.05,1.,0.05)
  numvects = 100; # Number of vectors to generate
  SNRdb = 20.;    # This is norm(signal)/norm(noise), so power, not energy
  # Values for lambda
  #lambdas = [0 10.^linspace(-5, 4, 10)];
  lambdas = numpy.array([0., 0.0001, 0.01, 1, 100, 10000])
  
  dosavedata = True
  savedataname = 'approx_pt_std2.mat'
  doshowplot = False
  dosaveplot = True
  saveplotbase = 'approx_pt_std2_'
  saveplotexts = ('png','pdf','eps')

  return algosN,algosL,d,sigma,deltas,rhos,lambdas,numvects,SNRdb,dosavedata,savedataname,\
          doshowplot,dosaveplot,saveplotbase,saveplotexts
  
  
  # Standard parameters 3
# All algorithms, 100 vectors
# d=50, sigma = 2, delta and rho full resolution (0.05 step), lambdas = 0, 1e-4, 1e-2, 1, 100, 10000
# Do save data, do save plots, don't show plots
# IDENTICAL with 1 but with 10dB SNR noise
def std3():
  # Define which algorithms to run
  algosN = gap,sl0analysis,bpanalysis,nesta        # tuple of algorithms not depending on lambda
  algosL = sl0,bp,ompeps,tst    # tuple of algorithms depending on lambda (our ABS approach)
  
  d = 50.0;
  sigma = 2.0
  deltas = numpy.arange(0.05,1.,0.05)
  rhos = numpy.arange(0.05,1.,0.05)
  numvects = 100; # Number of vectors to generate
  SNRdb = 10.;    # This is norm(signal)/norm(noise), so power, not energy
  # Values for lambda
  #lambdas = [0 10.^linspace(-5, 4, 10)];
  lambdas = numpy.array([0., 0.0001, 0.01, 1, 100, 10000])
  
  dosavedata = True
  savedataname = 'approx_pt_std3.mat'
  doshowplot = False
  dosaveplot = True
  saveplotbase = 'approx_pt_std3_'
  saveplotexts = ('png','pdf','eps')

  return algosN,algosL,d,sigma,deltas,rhos,lambdas,numvects,SNRdb,dosavedata,savedataname,\
          doshowplot,dosaveplot,saveplotbase,saveplotexts
          
# Standard parameters 4
# All algorithms, 100 vectors
# d=20, sigma = 10, delta and rho full resolution (0.05 step), lambdas = 0, 1e-4, 1e-2, 1, 100, 10000
# Do save data, do save plots, don't show plots
# Identical to 2 but with 10dB SNR noise
def std4():
  # Define which algorithms to run
  algosN = gap,sl0analysis,bpanalysis,nesta    # tuple of algorithms not depending on lambda
  algosL = sl0,bp,ompeps,tst    # tuple of algorithms depending on lambda (our ABS approach)
  
  d = 20.0
  sigma = 10.0
  deltas = numpy.arange(0.05,1.,0.05)
  rhos = numpy.arange(0.05,1.,0.05)
  numvects = 100; # Number of vectors to generate
  SNRdb = 10.;    # This is norm(signal)/norm(noise), so power, not energy
  # Values for lambda
  #lambdas = [0 10.^linspace(-5, 4, 10)];
  lambdas = numpy.array([0., 0.0001, 0.01, 1, 100, 10000])
  
  dosavedata = True
  savedataname = 'approx_pt_std4.mat'
  doshowplot = False
  dosaveplot = True
  saveplotbase = 'approx_pt_std4_'
  saveplotexts = ('png','pdf','eps')

  return algosN,algosL,d,sigma,deltas,rhos,lambdas,numvects,SNRdb,dosavedata,savedataname,\
          doshowplot,dosaveplot,saveplotbase,saveplotexts
          
# Standard parameters 1nesta
# Only NESTA, 100 vectors
# d=50, sigma = 2, delta and rho full resolution (0.05 step), lambdas = 0, 1e-4, 1e-2, 1, 100, 10000
# Do save data, do save plots, don't show plots
# Identical to std1 but with only NESTA
def std1nesta():
  # Define which algorithms to run
  algosN = nesta,               # tuple of algorithms not depending on lambda
  algosL = ()    # tuple of algorithms depending on lambda (our ABS approach)
  
  d = 50.0;
  sigma = 2.0
  deltas = numpy.arange(0.05,1.,0.05)
  rhos = numpy.arange(0.05,1.,0.05)
  numvects = 100; # Number of vectors to generate
  SNRdb = 20.;    # This is norm(signal)/norm(noise), so power, not energy
  # Values for lambda
  #lambdas = [0 10.^linspace(-5, 4, 10)];
  lambdas = numpy.array([0., 0.0001, 0.01, 1, 100, 10000])
  
  dosavedata = True
  savedataname = 'approx_pt_std1nesta.mat'
  doshowplot = False
  dosaveplot = True
  saveplotbase = 'approx_pt_std1nesta_'
  saveplotexts = ('png','pdf','eps')

  return algosN,algosL,d,sigma,deltas,rhos,lambdas,numvects,SNRdb,dosavedata,savedataname,\
          doshowplot,dosaveplot,saveplotbase,saveplotexts    
          
# Standard parameters 2nesta
# Only NESTA, 100 vectors
# d=20, sigma = 10, delta and rho full resolution (0.05 step), lambdas = 0, 1e-4, 1e-2, 1, 100, 10000
# Do save data, do save plots, don't show plots
# Identical with std2, but with only NESTA
def std2nesta():
  # Define which algorithms to run
  algosN = nesta,      # tuple of algorithms not depending on lambda
  algosL = ()    # tuple of algorithms depending on lambda (our ABS approach)
  
  d = 20.0
  sigma = 10.0
  deltas = numpy.arange(0.05,1.,0.05)
  rhos = numpy.arange(0.05,1.,0.05)
  numvects = 100; # Number of vectors to generate
  SNRdb = 20.;    # This is norm(signal)/norm(noise), so power, not energy
  # Values for lambda
  #lambdas = [0 10.^linspace(-5, 4, 10)];
  lambdas = numpy.array([0., 0.0001, 0.01, 1, 100, 10000])
  
  dosavedata = True
  savedataname = 'approx_pt_std2nesta.mat'
  doshowplot = False
  dosaveplot = True
  saveplotbase = 'approx_pt_std2nesta_'
  saveplotexts = ('png','pdf','eps')

  return algosN,algosL,d,sigma,deltas,rhos,lambdas,numvects,SNRdb,dosavedata,savedataname,\
          doshowplot,dosaveplot,saveplotbase,saveplotexts

  # Standard parameters 3nesta
# Only NESTA, 100 vectors
# d=50, sigma = 2, delta and rho full resolution (0.05 step), lambdas = 0, 1e-4, 1e-2, 1, 100, 10000
# Do save data, do save plots, don't show plots
# IDENTICAL with 3 but with only NESTA
def std3nesta():
  # Define which algorithms to run
  algosN = nesta,               # tuple of algorithms not depending on lambda
  algosL = ()    # tuple of algorithms depending on lambda (our ABS approach)
  
  d = 50.0;
  sigma = 2.0
  deltas = numpy.arange(0.05,1.,0.05)
  rhos = numpy.arange(0.05,1.,0.05)
  numvects = 100; # Number of vectors to generate
  SNRdb = 10.;    # This is norm(signal)/norm(noise), so power, not energy
  # Values for lambda
  #lambdas = [0 10.^linspace(-5, 4, 10)];
  lambdas = numpy.array([0., 0.0001, 0.01, 1, 100, 10000])
  
  dosavedata = True
  savedataname = 'approx_pt_std3nesta.mat'
  doshowplot = False
  dosaveplot = True
  saveplotbase = 'approx_pt_std3nesta_'
  saveplotexts = ('png','pdf','eps')

  return algosN,algosL,d,sigma,deltas,rhos,lambdas,numvects,SNRdb,dosavedata,savedataname,\
          doshowplot,dosaveplot,saveplotbase,saveplotexts
          
# Standard parameters 4nesta
# Only NESTA, 100 vectors
# d=20, sigma = 10, delta and rho full resolution (0.05 step), lambdas = 0, 1e-4, 1e-2, 1, 100, 10000
# Do save data, do save plots, don't show plots
# Identical to 4 but with only NESTA
def std4nesta():
  # Define which algorithms to run
  algosN = nesta,      # tuple of algorithms not depending on lambda
  algosL = ()    # tuple of algorithms depending on lambda (our ABS approach)
  
  d = 20.0
  sigma = 10.0
  deltas = numpy.arange(0.05,1.,0.05)
  rhos = numpy.arange(0.05,1.,0.05)
  numvects = 100; # Number of vectors to generate
  SNRdb = 10.;    # This is norm(signal)/norm(noise), so power, not energy
  # Values for lambda
  #lambdas = [0 10.^linspace(-5, 4, 10)];
  lambdas = numpy.array([0., 0.0001, 0.01, 1, 100, 10000])
  
  dosavedata = True
  savedataname = 'approx_pt_std4nesta.mat'
  doshowplot = False
  dosaveplot = True
  saveplotbase = 'approx_pt_std4nesta_'
  saveplotexts = ('png','pdf','eps')

  return algosN,algosL,d,sigma,deltas,rhos,lambdas,numvects,SNRdb,dosavedata,savedataname,\
          doshowplot,dosaveplot,saveplotbase,saveplotexts