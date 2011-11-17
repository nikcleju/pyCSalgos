# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 18:08:40 2011

@author: Nic
"""

import numpy as np
import scipy.io
import math
import os

import pyCSalgos
import pyCSalgos.GAP.GAP
import pyCSalgos.BP.l1qc
import pyCSalgos.BP.l1qec
import pyCSalgos.SL0.SL0_approx
import pyCSalgos.OMP.omp_QR
import pyCSalgos.RecomTST.RecommendedTST

#==========================
# Algorithm functions
#==========================
def run_gap(y,M,Omega,epsilon):
  gapparams = {"num_iteration" : 1000,\
                   "greedy_level" : 0.9,\
                   "stopping_coefficient_size" : 1e-4,\
                   "l2solver" : 'pseudoinverse',\
                   "noise_level": epsilon}
  return pyCSalgos.GAP.GAP.GAP(y,M,M.T,Omega,Omega.T,gapparams,np.zeros(Omega.shape[1]))[0]

def run_bp_analysis(y,M,Omega,epsilon):
  
  N,n = Omega.shape
  D = np.linalg.pinv(Omega)
  U,S,Vt = np.linalg.svd(D)
  Aeps = np.dot(M,D)
  Aexact = Vt[-(N-n):,:]
  # We don't ned any aggregate matrices anymore
  
  x0 = np.zeros(N)
  return np.dot(D , pyCSalgos.BP.l1qec.l1qec_logbarrier(x0,Aeps,Aeps.T,y,epsilon,Aexact,Aexact.T,np.zeros(N-n)))

def run_sl0_analysis(y,M,Omega,epsilon):
  
  N,n = Omega.shape
  D = np.linalg.pinv(Omega)
  U,S,Vt = np.linalg.svd(D)
  Aeps = np.dot(M,D)
  Aexact = Vt[-(N-n):,:]
  # We don't ned any aggregate matrices anymore
  
  sigmamin = 0.001
  sigma_decrease_factor = 0.5
  mu_0 = 2
  L = 10
  return np.dot(D , pyCSalgos.SL0.SL0_approx.SL0_approx_analysis(Aeps,Aexact,y,epsilon,sigmamin,sigma_decrease_factor,mu_0,L))

def run_sl0(y,M,Omega,D,U,S,Vt,epsilon,lbd):
  
  N,n = Omega.shape
  #D = np.linalg.pinv(Omega)
  #U,S,Vt = np.linalg.svd(D)
  aggDupper = np.dot(M,D)
  aggDlower = Vt[-(N-n):,:]
  aggD = np.concatenate((aggDupper, lbd * aggDlower))
  aggy = np.concatenate((y, np.zeros(N-n)))
  
  sigmamin = 0.001
  sigma_decrease_factor = 0.5
  mu_0 = 2
  L = 10
  return pyCSalgos.SL0.SL0_approx.SL0_approx(aggD,aggy,epsilon,sigmamin,sigma_decrease_factor,mu_0,L)
  
def run_bp(y,M,Omega,D,U,S,Vt,epsilon,lbd):
  
  N,n = Omega.shape
  #D = np.linalg.pinv(Omega)
  #U,S,Vt = np.linalg.svd(D)
  aggDupper = np.dot(M,D)
  aggDlower = Vt[-(N-n):,:]
  aggD = np.concatenate((aggDupper, lbd * aggDlower))
  aggy = np.concatenate((y, np.zeros(N-n)))

  x0 = np.zeros(N)
  return pyCSalgos.BP.l1qc.l1qc_logbarrier(x0,aggD,aggD.T,aggy,epsilon)

def run_ompeps(y,M,Omega,D,U,S,Vt,epsilon,lbd):
  
  N,n = Omega.shape
  #D = np.linalg.pinv(Omega)
  #U,S,Vt = np.linalg.svd(D)
  aggDupper = np.dot(M,D)
  aggDlower = Vt[-(N-n):,:]
  aggD = np.concatenate((aggDupper, lbd * aggDlower))
  aggy = np.concatenate((y, np.zeros(N-n)))
  
  opts = dict()
  opts['stopCrit'] = 'mse'
  opts['stopTol'] = epsilon**2 / aggy.size
  return pyCSalgos.OMP.omp_QR.greed_omp_qr(aggy,aggD,aggD.shape[1],opts)[0]
  
def run_tst(y,M,Omega,D,U,S,Vt,epsilon,lbd):
  
  N,n = Omega.shape
  #D = np.linalg.pinv(Omega)
  #U,S,Vt = np.linalg.svd(D)
  aggDupper = np.dot(M,D)
  aggDlower = Vt[-(N-n):,:]
  aggD = np.concatenate((aggDupper, lbd * aggDlower))
  aggy = np.concatenate((y, np.zeros(N-n)))
  
  nsweep = 300
  tol = epsilon / np.linalg.norm(aggy)
  return pyCSalgos.RecomTST.RecommendedTST.RecommendedTST(aggD, aggy, nsweep=nsweep, tol=tol)

#==========================
# Define tuples (algorithm function, name)
#==========================
gap = (run_gap, 'GAP')
sl0 = (run_sl0, 'SL0a')
sl0analysis = (run_sl0_analysis, 'SL0a2')
bpanalysis = (run_bp_analysis, 'BPa2')
bp  = (run_bp, 'BP')
ompeps = (run_ompeps, 'OMPeps')
tst = (run_tst, 'TST')

#==========================
# Pool initializer function (multiprocessing)
# Needed to pass the shared variable to the worker processes
# The variables must be global in the module in order to be seen later in run_once_tuple()
# see http://stackoverflow.com/questions/1675766/how-to-combine-pool-map-with-array-shared-memory-in-python-multiprocessing
#==========================
def initProcess(share, njobs):
    import sys
    currmodule = sys.modules[__name__]
    currmodule.proccount = share
    currmodule.njobs = njobs
  
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
  algosN = gap,      # tuple of algorithms not depending on lambda
  algosL = sl0,bp    # tuple of algorithms depending on lambda (our ABS approach)
  
  d = 50.0
  sigma = 2.0
  deltas = np.array([0.05, 0.45, 0.95])
  rhos = np.array([0.05, 0.45, 0.95])
  numvects = 10; # Number of vectors to generate
  SNRdb = 20.;    # This is norm(signal)/norm(noise), so power, not energy
  # Values for lambda
  #lambdas = [0 10.^linspace(-5, 4, 10)];
  lambdas = np.array([0., 0.0001, 0.01, 1, 100, 10000])
  
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
  algosN = gap,sl0analysis,bpanalysis               # tuple of algorithms not depending on lambda
  algosL = sl0,bp,ompeps,tst    # tuple of algorithms depending on lambda (our ABS approach)
  
  d = 50.0;
  sigma = 2.0
  deltas = np.arange(0.05,1.,0.05)
  rhos = np.arange(0.05,1.,0.05)
  numvects = 100; # Number of vectors to generate
  SNRdb = 20.;    # This is norm(signal)/norm(noise), so power, not energy
  # Values for lambda
  #lambdas = [0 10.^linspace(-5, 4, 10)];
  lambdas = np.array([0., 0.0001, 0.01, 1, 100, 10000])
  
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
  algosN = gap,sl0analysis,bpanalysis      # tuple of algorithms not depending on lambda
  algosL = sl0,bp,ompeps,tst    # tuple of algorithms depending on lambda (our ABS approach)
  
  d = 20.0
  sigma = 10.0
  deltas = np.arange(0.05,1.,0.05)
  rhos = np.arange(0.05,1.,0.05)
  numvects = 100; # Number of vectors to generate
  SNRdb = 20.;    # This is norm(signal)/norm(noise), so power, not energy
  # Values for lambda
  #lambdas = [0 10.^linspace(-5, 4, 10)];
  lambdas = np.array([0., 0.0001, 0.01, 1, 100, 10000])
  
  dosavedata = True
  savedataname = 'approx_pt_std2.mat'
  doshowplot = False
  dosaveplot = True
  saveplotbase = 'approx_pt_std2_'
  saveplotexts = ('png','pdf','eps')

  return algosN,algosL,d,sigma,deltas,rhos,lambdas,numvects,SNRdb,dosavedata,savedataname,\
          doshowplot,dosaveplot,saveplotbase,saveplotexts
  
#==========================
# Interface run functions
#==========================
def run_mp(std=std2,ncpus=None):
  
  algosN,algosL,d,sigma,deltas,rhos,lambdas,numvects,SNRdb,dosavedata,savedataname,doshowplot,dosaveplot,saveplotbase,saveplotexts = std()
  run_multi(algosN, algosL, d,sigma,deltas,rhos,lambdas,numvects,SNRdb,dosavedata=dosavedata,savedataname=savedataname,\
            doparallel=True, ncpus=ncpus,\
            doshowplot=doshowplot,dosaveplot=dosaveplot,saveplotbase=saveplotbase,saveplotexts=saveplotexts)

def run(std=std2):
  algosN,algosL,d,sigma,deltas,rhos,lambdas,numvects,SNRdb,dosavedata,savedataname,doshowplot,dosaveplot,saveplotbase,saveplotexts = std()
  run_multi(algosN, algosL, d,sigma,deltas,rhos,lambdas,numvects,SNRdb,dosavedata=dosavedata,savedataname=savedataname,\
            doparallel=False,\
            doshowplot=doshowplot,dosaveplot=dosaveplot,saveplotbase=saveplotbase,saveplotexts=saveplotexts)  
#==========================
# Main functions  
#==========================
def run_multi(algosN, algosL, d, sigma, deltas, rhos, lambdas, numvects, SNRdb,
            doparallel=False, ncpus=None,\
            doshowplot=False, dosaveplot=False, saveplotbase=None, saveplotexts=None,\
            dosavedata=False, savedataname=None):

  print "This is analysis recovery ABS approximation script by Nic"
  print "Running phase transition ( run_multi() )"

  # Not only for parallel  
  #if doparallel:
  import multiprocessing
  # Shared value holding the number of finished processes
  # Add it as global of the module
  import sys
  currmodule = sys.modules[__name__]
  currmodule.proccount = multiprocessing.Value('I', 0) # 'I' = unsigned int, see docs (multiprocessing, array)
    
  if dosaveplot or doshowplot:
    try:
      import matplotlib
      if doshowplot or os.name == 'nt':
        print "Importing matplotlib with default (GUI) backend... ",
      else:
        print "Importing matplotlib with \"Cairo\" backend... ",
        matplotlib.use('Cairo')
      import matplotlib.pyplot as plt
      import matplotlib.cm as cm
      print "OK"        
    except:
      print "FAIL"
      print "Importing matplotlib.pyplot failed. No figures at all"
      print "Try selecting a different backend"
      doshowplot = False
      dosaveplot = False
  
  # Print summary of parameters
  print "Parameters:"
  if doparallel:
    if ncpus is None:
      print "  Running in parallel with default threads using \"multiprocessing\" package"
    else:
      print "  Running in parallel with",ncpus,"threads using \"multiprocessing\" package"
  else:
    print "Running single thread"
  if doshowplot:
    print "  Showing figures"
  else:
    print "  Not showing figures"
  if dosaveplot:
    print "  Saving figures as "+saveplotbase+"* with extensions ",saveplotexts
  else:
    print "  Not saving figures"
  print "  Running algorithms",[algotuple[1] for algotuple in algosN],[algotuple[1] for algotuple in algosL]
  
  nalgosN = len(algosN)  
  nalgosL = len(algosL)
  
  meanmatrix = dict()
  for i,algo in zip(np.arange(nalgosN),algosN):
    meanmatrix[algo[1]]   = np.zeros((rhos.size, deltas.size))
  for i,algo in zip(np.arange(nalgosL),algosL):
    meanmatrix[algo[1]]   = np.zeros((lambdas.size, rhos.size, deltas.size))
  
  # Prepare parameters
  jobparams = []
  print "  (delta, rho) pairs to be run:"
  for idelta,delta in zip(np.arange(deltas.size),deltas):
    for irho,rho in zip(np.arange(rhos.size),rhos):
      
      # Generate data and operator
      Omega,x0,y,M,realnoise = generateData(d,sigma,delta,rho,numvects,SNRdb)
      
      #Save the parameters, and run after
      print "    delta = ",delta," rho = ",rho
      jobparams.append((algosN,algosL, Omega,y,lambdas,realnoise,M,x0))
  
  # Not only for parallel
  #if doparallel:
  currmodule.njobs = deltas.size * rhos.size  
  print "End of parameters"
  
  # Run
  jobresults = []
  
  if doparallel:
    pool = multiprocessing.Pool(4,initializer=initProcess,initargs=(currmodule.proccount,currmodule.njobs))
    jobresults = pool.map(run_once_tuple, jobparams)
  else:
    for jobparam in jobparams:
      jobresults.append(run_once_tuple(jobparam))

  # Read results
  idx = 0
  for idelta,delta in zip(np.arange(deltas.size),deltas):
    for irho,rho in zip(np.arange(rhos.size),rhos):
      mrelerrN,mrelerrL = jobresults[idx]
      idx = idx+1
      for algotuple in algosN: 
        meanmatrix[algotuple[1]][irho,idelta] = 1 - mrelerrN[algotuple[1]]
        if meanmatrix[algotuple[1]][irho,idelta] < 0 or math.isnan(meanmatrix[algotuple[1]][irho,idelta]):
          meanmatrix[algotuple[1]][irho,idelta] = 0
      for algotuple in algosL:
        for ilbd in np.arange(lambdas.size):
          meanmatrix[algotuple[1]][ilbd,irho,idelta] = 1 - mrelerrL[algotuple[1]][ilbd]
          if meanmatrix[algotuple[1]][ilbd,irho,idelta] < 0 or math.isnan(meanmatrix[algotuple[1]][ilbd,irho,idelta]):
            meanmatrix[algotuple[1]][ilbd,irho,idelta] = 0

  # Save
  if dosavedata:
    tosave = dict()
    tosave['meanmatrix'] = meanmatrix
    tosave['d'] = d
    tosave['sigma'] = sigma
    tosave['deltas'] = deltas
    tosave['rhos'] = rhos
    tosave['numvects'] = numvects
    tosave['SNRdb'] = SNRdb
    tosave['lambdas'] = lambdas
    # Save algo names as cell array
    obj_arr = np.zeros((len(algosN)+len(algosL),), dtype=np.object)
    idx = 0
    for algotuple in algosN+algosL:
      obj_arr[idx] = algotuple[1]
      idx = idx+1
    tosave['algonames'] = obj_arr
    try:
      scipy.io.savemat(savedataname, tosave)
    except:
      print "Save error"
  # Show
  if doshowplot or dosaveplot:
    for algotuple in algosN:
      algoname = algotuple[1]
      plt.figure()
      plt.imshow(meanmatrix[algoname], cmap=cm.gray, interpolation='nearest',origin='lower')
      if dosaveplot:
        for ext in saveplotexts:
          plt.savefig(saveplotbase + algoname + '.' + ext, bbox_inches='tight')
    for algotuple in algosL:
      algoname = algotuple[1]
      for ilbd in np.arange(lambdas.size):
        plt.figure()
        plt.imshow(meanmatrix[algoname][ilbd], cmap=cm.gray, interpolation='nearest',origin='lower')
        if dosaveplot:
          for ext in saveplotexts:
            plt.savefig(saveplotbase + algoname + ('_lbd%.0e' % lambdas[ilbd]) + '.' + ext, bbox_inches='tight')
    if doshowplot:
      plt.show()
    
  print "Finished."
  
def run_once_tuple(t):
  results = run_once(*t)
  import sys
  currmodule = sys.modules[__name__]  
  currmodule.proccount.value = currmodule.proccount.value + 1
  print "================================"
  print "Finished job",currmodule.proccount.value,"of",currmodule.njobs
  print "================================"
  return results

def run_once(algosN,algosL,Omega,y,lambdas,realnoise,M,x0):
  
  d = Omega.shape[1]  
  
  nalgosN = len(algosN)  
  nalgosL = len(algosL)
  
  
  xrec = dict()
  err = dict()
  relerr = dict()

  # Prepare storage variables for algorithms non-Lambda
  for i,algo in zip(np.arange(nalgosN),algosN):
    xrec[algo[1]]   = np.zeros((d, y.shape[1]))
    err[algo[1]]    = np.zeros(y.shape[1])
    relerr[algo[1]] = np.zeros(y.shape[1])
  # Prepare storage variables for algorithms with Lambda    
  for i,algo in zip(np.arange(nalgosL),algosL):
    xrec[algo[1]]   = np.zeros((lambdas.size, d, y.shape[1]))
    err[algo[1]]    = np.zeros((lambdas.size, y.shape[1]))
    relerr[algo[1]] = np.zeros((lambdas.size, y.shape[1]))
  
  # Run algorithms non-Lambda
  for iy in np.arange(y.shape[1]):
    for algofunc,strname in algosN:
      epsilon = 1.1 * np.linalg.norm(realnoise[:,iy])
      try:
        xrec[strname][:,iy] = algofunc(y[:,iy],M,Omega,epsilon)
      except pyCSalgos.BP.l1qec.l1qecInputValueError as e:
        print "Caught exception when running algorithm",strname," :",e.message
      err[strname][iy]    = np.linalg.norm(x0[:,iy] - xrec[strname][:,iy])
      relerr[strname][iy] = err[strname][iy] / np.linalg.norm(x0[:,iy])
  for algofunc,strname in algosN:
    print strname,' : avg relative error = ',np.mean(relerr[strname])  

  # Run algorithms with Lambda
  for ilbd,lbd in zip(np.arange(lambdas.size),lambdas):
    for iy in np.arange(y.shape[1]):
      D = np.linalg.pinv(Omega)
      U,S,Vt = np.linalg.svd(D)
      for algofunc,strname in algosL:
        epsilon = 1.1 * np.linalg.norm(realnoise[:,iy])
        try:
          gamma = algofunc(y[:,iy],M,Omega,D,U,S,Vt,epsilon,lbd)
        except pyCSalgos.BP.l1qc.l1qcInputValueError as e:
          print "Caught exception when running algorithm",strname," :",e.message
        xrec[strname][ilbd,:,iy] = np.dot(D,gamma)
        err[strname][ilbd,iy]    = np.linalg.norm(x0[:,iy] - xrec[strname][ilbd,:,iy])
        relerr[strname][ilbd,iy] = err[strname][ilbd,iy] / np.linalg.norm(x0[:,iy])
    print 'Lambda = ',lbd,' :'
    for algofunc,strname in algosL:
      print '   ',strname,' : avg relative error = ',np.mean(relerr[strname][ilbd,:])

  # Prepare results
  mrelerrN = dict()
  for algotuple in algosN:
    mrelerrN[algotuple[1]] = np.mean(relerr[algotuple[1]])
  mrelerrL = dict()
  for algotuple in algosL:
    mrelerrL[algotuple[1]] = np.mean(relerr[algotuple[1]],1)
  
  return mrelerrN,mrelerrL

def generateData(d,sigma,delta,rho,numvects,SNRdb):

  # Process parameters
  noiselevel = 1.0 / (10.0**(SNRdb/10.0));
  p = round(sigma*d);
  m = round(delta*d);
  l = round(d - rho*m);
  
  # Generate Omega and data based on parameters
  Omega = pyCSalgos.GAP.GAP.Generate_Analysis_Operator(d, p);
  # Optionally make Omega more coherent
  U,S,Vt = np.linalg.svd(Omega);
  Sdnew = S * (1+np.arange(S.size)) # Make D coherent, not Omega!
  Snew = np.vstack((np.diag(Sdnew), np.zeros((Omega.shape[0] - Omega.shape[1], Omega.shape[1]))))
  Omega = np.dot(U , np.dot(Snew,Vt))

  # Generate data  
  x0,y,M,Lambda,realnoise = pyCSalgos.GAP.GAP.Generate_Data_Known_Omega(Omega, d,p,m,l,noiselevel, numvects,'l0');
  
  return Omega,x0,y,M,realnoise
  
# Script main
if __name__ == "__main__":
  #import cProfile
  #cProfile.run('mainrun()', 'profile')    
  run(std3)
