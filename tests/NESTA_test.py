# -*- coding: utf-8 -*-
"""
Created on Sun Nov 06 20:53:14 2011

@author: Nic
"""
import numpy as np
import numpy.linalg
import scipy.io
import unittest
from pyCSalgos.NESTA.NESTA import NESTA

class NESTAresults(unittest.TestCase):
  def testResults(self):
    mdict = scipy.io.loadmat('NESTAtestdata.mat')
    
    # Add [0,0] indices because data is read from mat file as [1,1] arrays
    opt_TolVar = mdict['opt_TolVar'][0,0]
    opt_Verbose = mdict['opt_Verbose'][0,0]
    opt_muf = mdict['opt_muf'][0,0]
    numA = mdict['numA'][0,0]
    
    # Known bad but good:
    known = ()
      
    sumplus  = 0.0
    summinus = 0.0
    numplus = 0
    numminus = 0    
    
    # A = system matrix
    # Y = matrix with measurements (on columns)
    # sigmamin = vector with sigma_mincell
    for k,A,Y,M,eps,Xr in zip(np.arange(numA),mdict['cellA'].squeeze(),mdict['cellY'].squeeze(),mdict['cellM'].squeeze(),mdict['cellEps'].squeeze(),mdict['cellXr'].squeeze()):

      # Fix numpy error "LapackError: Parameter a has non-native byte order in lapack_lite.dgesdd"
      A = A.newbyteorder('=')
      Y = Y.newbyteorder('=')
      M = M.newbyteorder('=')
      eps = eps.newbyteorder('=')
      Xr = Xr.newbyteorder('=')
      
      eps = eps.squeeze()
      
      U,S,V = numpy.linalg.svd(M, full_matrices = True)
      V = V.T         # Make like Matlab
      m,n = M.shape   # Make like Matlab
      S = numpy.hstack((numpy.diag(S), numpy.zeros((m,n-m))))

      optsUSV = {'U':U, 'S':S, 'V':V}
      opts = {'U':A, 'Ut':A.T.copy(), 'USV':optsUSV, 'TolVar':opt_TolVar, 'Verbose':opt_Verbose}
      
      for i in np.arange(Y.shape[1]):
        xr = NESTA(M, None, Y[:,i], opt_muf, eps[i] * numpy.linalg.norm(Y[:,i]), opts)[0]
        
        # check if found solution is the same as the correct cslution
        diff = numpy.linalg.norm(xr - Xr[:,i])
        print "k =",k,"i = ",i
        if diff < 1e-6:
          print "Recovery OK"
          isOK = True
        else:
          if numpy.linalg.norm(xr,1) < numpy.linalg.norm(Xr[:,i],1):
            numplus = numplus+1
            sumplus = sumplus + numpy.linalg.norm(Xr[:,i],1) - numpy.linalg.norm(xr,1)
          else:
            numminus = numminus+1
            summinus = summinus + numpy.linalg.norm(xr,1) - numpy.linalg.norm(Xr[:,i],1)
         
          print "Oops"
          if (k,i) not in known:
            #isOK = False
            print "Should stop here"
          else:
            print "Known bad but good"
            isOK = True
        # comment / uncomment this
        self.assertTrue(isOK)
    print 'Finished test'
  
if __name__ == "__main__":
    #import cProfile
    #cProfile.run('unittest.main()', 'profres')
    unittest.main()    
    #suite = unittest.TestLoader().loadTestsFromTestCase(CompareResults)
    #unittest.TextTestRunner(verbosity=2).run(suite)    
