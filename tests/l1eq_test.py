# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 14:28:08 2011

@author: Nic

Test l1-magic l1eq_pd() algorithm
"""

import numpy as np
import numpy.linalg
import scipy.io
import unittest
from pyCSalgos.BP.l1eq_pd import l1eq_pd
#from l1qc import l1qc_logbarrier

class l1eq_results(unittest.TestCase):
  def testResults(self):
    mdict = scipy.io.loadmat('l1eq_testdata.mat')
    
    # A = system matrix
    # Y = matrix with measurements (on columns)
    # X0 = matrix with initial solutions (on columns)
    # Xr = matrix with correct solutions (on columns)
    for A,Y,X0,Xr in zip(mdict['cellA'].squeeze(),mdict['cellY'].squeeze(),mdict['X0'].squeeze(),mdict['Xr'].squeeze()):
      for i in np.arange(Y.shape[1]):
        xr = l1eq_pd(X0[:,i], A, np.array([]), Y[:,i])
        
        # check if found solution is the same as the correct cslution
        diff = numpy.linalg.norm(xr - Xr[:,i])
        err1 = numpy.linalg.norm(Y[:,i] - np.dot(A,xr))
        err2 = numpy.linalg.norm(Y[:,i] - np.dot(A,Xr[:,i]))
        norm1 = numpy.linalg.norm(xr,1)
        norm2 = numpy.linalg.norm(Xr[:,i],1)
        print 'diff = ',diff
        print 'err1 = ',err1
        print 'err2 = ',err2
        print 'norm1 = ',norm1
        print 'norm2 = ',norm2
        
        # It seems Matlab's linsolve and scipy solve are slightly different
        # Therefore make a more robust condition:
        #  OK;    if   solutions are close enough (diff < 1e-6)
        #              or
        #              (
        #               they fulfill the constraint close enough (differr < 1e-6)
        #                 and
        #               Python solution has l1 norm no more than 1e-6 larger as the reference solution
        #                 (i.e. either norm1 < norm2   or   norm1>norm2 not by more than 1e-6)
        #              )
        #        
        #  ERROR: else
        differr  = abs((err1 - err2))
        diffnorm = norm1 - norm2  # intentionately no abs(), since norm1 < norm2 is good
        if diff < 1e-6 or (differr < 1e-6 and (diffnorm < 1e-6)):
          isok = True
        else:
          isok = False

        #if not isok:
        #  print "should raise"
        #  #self.assertTrue(isok)
        self.assertTrue(isok)
  
if __name__ == "__main__":
    unittest.main(verbosity=2)    
    #suite = unittest.TestLoader().loadTestsFromTestCase(CompareResults)
    #unittest.TextTestRunner(verbosity=2).run(suite)    