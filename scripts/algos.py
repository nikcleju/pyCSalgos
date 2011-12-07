# -*- coding: utf-8 -*-
"""
Created on Wed Dec 07 14:06:13 2011

@author: ncleju
"""

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

def run_nesta(y,M,Omega,epsilon):
  
  U,S,V = np.linalg.svd(M, full_matrices = True)
  V = V.T         # Make like Matlab
  m,n = M.shape   # Make like Matlab
  S = np.hstack((np.diag(S), np.zeros((m,n-m))))  

  opt_muf = 1e-3
  optsUSV = {'U':U, 'S':S, 'V':V}
  opts = {'U':Omega, 'Ut':Omega.T.copy(), 'USV':optsUSV, 'TolVar':1e-5, 'Verbose':0}
  return pyCSalgos.NESTA.NESTA.NESTA(M, None, y, opt_muf, epsilon, opts)[0]


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
nesta = (run_nesta, 'NESTA')
bp  = (run_bp, 'BP')
ompeps = (run_ompeps, 'OMPeps')
tst = (run_tst, 'TST')