# -*- coding: utf-8 -*-
"""
Created on Sat Nov 05 21:34:49 2011

@author: Nic
"""
import numpy as np
import numpy.linalg
import scipy.io
import unittest
from pyCSalgos.SL0.SL0_approx import SL0_approx

class SL0results(unittest.TestCase):
  def testResults(self):
    mdict = scipy.io.loadmat('SL0approxtestdata.mat')
    
    # A = system matrix
    # Y = matrix with measurements (on columns)
    # sigmamin = vector with sigma_min
    for A,Y,eps,sigmamin,Xr in zip(mdict['cellA'].squeeze(),mdict['cellY'].squeeze(),mdict['cellEps'].squeeze(),mdict['sigmamin'].squeeze(),mdict['cellXr'].squeeze()):
      for i in np.arange(Y.shape[1]):
        
        # Fix numpy error "LapackError: Parameter a has non-native byte order in lapack_lite.dgesdd"
        A = A.newbyteorder('=')
        Y = Y.newbyteorder('=')
        eps = eps.newbyteorder('=')
        sigmamin = sigmamin.newbyteorder('=')
        Xr = Xr.newbyteorder('=')
        
        xr = SL0_approx(A, Y[:,i], eps.squeeze()[i], sigmamin)
        
        # check if found solution is the same as the correct cslution
        diff = numpy.linalg.norm(xr - Xr[:,i])
        self.assertTrue(diff < 1e-12)
    #        err1 = numpy.linalg.norm(Y[:,i] - np.dot(A,xr))
    #        err2 = numpy.linalg.norm(Y[:,i] - np.dot(A,Xr[:,i]))
    #        norm1 = numpy.linalg.norm(xr,1)
    #        norm2 = numpy.linalg.norm(Xr[:,i],1)
    #                
    #        # Make a more robust condition:
    #        #  OK;    if   solutions are close enough (diff < 1e-6)
    #        #              or
    #        #              (
    #        #               Python solution fulfills the constraint better (or up to 1e-6 worse)
    #        #                 and
    #        #               Python solution has l1 norm no more than 1e-6 larger as the reference solution
    #        #                 (i.e. either norm1 < norm2   or   norm1>norm2 not by more than 1e-6)
    #        #              )
    #        #        
    #        #  ERROR: else        
    #        differr  = err1 - err2    # intentionately no abs(), since err1` < err2 is good
    #        diffnorm = norm1 - norm2  # intentionately no abs(), since norm1 < norm2 is good
    #        if diff < 1e-6 or (differr < 1e-6 and (diffnorm < 1e-6)):
    #          isok = True
    #        else:
    #          isok = False
    #        self.assertTrue(isok)
        
        #diff = numpy.linalg.norm(xr - Xr[:,i])
        #if diff > 1e-6:
        #    self.assertTrue(diff < 1e-6)

  
if __name__ == "__main__":
    #import cProfile
    #cProfile.run('unittest.main()', 'profres')
    unittest.main()    
    #suite = unittest.TestLoader().loadTestsFromTestCase(CompareResults)
    #unittest.TextTestRunner(verbosity=2).run(suite)    
