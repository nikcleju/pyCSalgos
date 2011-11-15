# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 12:28:55 2011

@author: ncleju
"""

import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def loadshowmatrices(filename, algonames = None):
    mdict = scipy.io.loadmat(filename)
    if algonames == None:
      algonames = mdict['algonames']
    
    for algonameobj in algonames:
        algoname = algonames[0][0]
        print algoname
        if mdict['meanmatrix'][algoname][0,0].ndim == 2:
            plt.figure()
            plt.imshow(mdict['meanmatrix'][algoname][0,0], cmap=cm.gray, interpolation='nearest',origin='lower')            
        elif mdict['meanmatrix'][algoname][0,0].ndim == 3:
            for matrix in mdict['meanmatrix'][algoname][0,0]:
                plt.figure()
                plt.imshow(matrix, cmap=cm.gray, interpolation='nearest',origin='lower')
    plt.show()
    print "Finished."
    
def loadsavematrices(filename, saveplotbase, saveplotexts, algonames = None):
    
    mdict = scipy.io.loadmat(filename)
    lambdas = mdict['lambdas']

    if algonames is None:
      algonames = mdict['algonames']
    
    for algonameobj in algonames:
        algoname = algonameobj[0][0]
        print algoname
        if mdict['meanmatrix'][algoname][0,0].ndim == 2:
            plt.figure()
            plt.imshow(mdict['meanmatrix'][algoname][0,0], cmap=cm.gray, interpolation='nearest',origin='lower')
            for ext in saveplotexts:
              plt.savefig(saveplotbase + algoname + '.' + ext, bbox_inches='tight')
        elif mdict['meanmatrix'][algoname][0,0].ndim == 3:
            ilbd = 0
            for matrix in mdict['meanmatrix'][algoname][0,0]:
                plt.figure()
                plt.imshow(matrix, cmap=cm.gray, interpolation='nearest',origin='lower')
                for ext in saveplotexts:
                  plt.savefig(saveplotbase + algoname + ('_lbd%.0e' % lambdas[ilbd]) + '.' + ext, bbox_inches='tight')
                ilbd = ilbd + 1
    print "Finished."    
            
def appendtomatfile(filename, toappend, toappendname):
  mdict = scipy.io.loadmat(filename)
  mdict[toappendname] = toappend
  try:
    scipy.io.savemat(filename, mdict)
  except:
    print "Save error"  
  
  # To save to a cell array, create an object array:
  #  >>> obj_arr = np.zeros((2,), dtype=np.object)
  #  >>> obj_arr[0] = 1
  #  >>> obj_arr[1] = 'a string'    