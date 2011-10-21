""" 
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
# Bob L. Sturm <bst@create.aau.dk> 20111018
# Department of Architecture, Design and Media Technology
# Aalborg University Copenhagen
# Lautrupvang 15, 2750 Ballerup, Denmark
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#
"""

import unittest

import numpy as np
from sklearn.utils import check_random_state
import time

from omp_sk_bugfix import orthogonal_mp
from omp_QR import greed_omp_qr
from omp_QR import omp_qr

"""
Run a problem suite involving sparse vectors in 
ambientDimension dimensional space, with a resolution
in the phase plane of numGradations x numGradations,
and at each indeterminacy and sparsity pair run
numTrials independent trials.

Outputs a text file denoting successes at each phase point.
For more on phase transitions, see:
D. L. Donoho and J. Tanner, "Precise undersampling theorems," 
Proc. IEEE, vol. 98, no. 6, pp. 913-924, June 2010.
"""

class CompareResults(unittest.TestCase):
    
    def testCompareResults(self):
      """OMP results should be almost the same with all implementations"""
      ambientDimension = 400
      numGradations = 30
      numTrials = 1
      runProblemSuite(ambientDimension,numGradations,numTrials, verbose=False) 



def runProblemSuite(ambientDimension,numGradations,numTrials, verbose):

  idx = np.arange(ambientDimension)
  phaseDelta = np.linspace(0.05,1,numGradations)
  phaseRho = np.linspace(0.05,1,numGradations)
  success = np.zeros((numGradations, numGradations))
  
  #Nic: init timers
  t1all = 0
  t2all = 0
  t3all = 0
  
  deltaCounter = 0
  # delta is number of measurements/
  for delta in phaseDelta[:17]:
    rhoCounter = 0
    for rho in phaseRho:
      if verbose:
          print(deltaCounter,rhoCounter)
          
      numMeasurements = int(delta*ambientDimension)
      sparsity = int(rho*numMeasurements)
      # how do I set the following to be random each time?
      generator = check_random_state(100)
      # create unit norm dictionary
      D = generator.randn(numMeasurements, ambientDimension)
      D /= np.sqrt(np.sum((D ** 2), axis=0))
      # compute Gramian (for efficiency)
      DTD = np.dot(D.T,D)
  
      successCounter = 0
      trial = numTrials
      while trial > 0:
        # generate sparse signal with a minimum non-zero value
        x = np.zeros((ambientDimension, 1))
        idx2 = idx
        generator.shuffle(idx2)
        idx3 = idx2[:sparsity]
        while np.min(np.abs(x[idx3,0])) < 1e-10 :
           x[idx3,0] = generator.randn(sparsity)
        # sense sparse signal
        y = np.dot(D, x)
        
        # Nic: Use sparsify OMP function (translated from Matlab)
        ompopts = dict({'stopCrit':'M', 'stopTol':2*sparsity})
        starttime = time.time()                     # start timer
        x_r2, errs, times = greed_omp_qr(y.squeeze().copy(), D.copy(), D.shape[1], ompopts)
        t2all = t2all + time.time() - starttime     # stop timer
        idx_r2 = np.nonzero(x_r2)[0]        
        
        # run to two times expected sparsity, or tolerance
        # why? Often times, OMP can retrieve the correct solution
        # when it is run for more than the expected sparsity
        #x_r, idx_r = omp_qr(y,D,DTD,2*sparsity,1e-5)
        # Nic: adjust tolerance to match with other function
        starttime = time.time()                     # start timer
        x_r, idx_r = omp_qr(y.copy(),D.copy(),DTD.copy(),2*sparsity,numMeasurements*1e-14/np.vdot(y,y))
        t1all = t1all + time.time() - starttime     # stop timer        
        
        # Nic: test sklearn omp
        starttime = time.time()                     # start timer
        x_r3 = orthogonal_mp(D.copy(), y.copy(), 2*sparsity, tol=numMeasurements*1e-14, precompute_gram=False, copy_X=True)
        idx_r3 = np.nonzero(x_r3)[0]
        t3all = t3all + time.time() - starttime     # stop timer        
        
        # Nic: compare results
        if verbose:
            print 'diff1 = ',np.linalg.norm(x_r.squeeze() - x_r2.squeeze())
            print 'diff2 = ',np.linalg.norm(x_r.squeeze() - x_r3.squeeze())
            print 'diff3 = ',np.linalg.norm(x_r2.squeeze() - x_r3.squeeze())
            print "Bob's total time = ", t1all
            print "Nic's total time = ", t2all
            print "Skl's total time = ", t3all
        if np.linalg.norm(x_r.squeeze() - x_r2.squeeze()) > 1e-6 or \
           np.linalg.norm(x_r.squeeze() - x_r3.squeeze()) > 1e-6 or \
           np.linalg.norm(x_r2.squeeze() - x_r3.squeeze()) > 1e-6:
               if verbose:
                   print "STOP: Different results"
                   print "Bob's residual: ||y - D x_r ||_2 = ",np.linalg.norm(y.squeeze() - np.dot(D,x_r).squeeze())
                   print "Nic's residual: ||y - D x_r ||_2 = ",np.linalg.norm(y.squeeze() - np.dot(D,x_r2).squeeze())
                   print "Skl's residual: ||y - D x_r ||_2 = ",np.linalg.norm(y.squeeze() - np.dot(D,x_r3).squeeze())
               raise ValueError("Different results")
  
        # debais to remove small entries
        for nn in idx_r:
          if abs(x_r[nn]) < 1e-10:
            x_r[nn] = 0
  
        # exact recovery condition using support
        #if sorted(np.flatnonzero(x_r)) == sorted(np.flatnonzero(x)):
        #  successCounter += 1
        # exact recovery condition using error in solution
        error = x - x_r
        """ the following is the exact recovery condition in: A. Maleki 
              and D. L. Donoho, "Optimally tuned iterative reconstruction 
              algorithms for compressed sensing," IEEE J. Selected Topics 
              in Signal Process., vol. 4, pp. 330-341, Apr. 2010. """
        if np.vdot(error,error) < np.vdot(x,x)*1e-4:
          successCounter += 1
        trial -= 1
  
      success[rhoCounter,deltaCounter] = successCounter
      if successCounter == 0:
        break
  
      rhoCounter += 1
      #np.savetxt('test.txt',success,fmt='#2.1d',delimiter=',')
    deltaCounter += 1
    
if __name__ == "__main__":
    
    unittest.main(verbosity=2)    
    #suite = unittest.TestLoader().loadTestsFromTestCase(CompareResults)
    #unittest.TextTestRunner(verbosity=2).run(suite)    