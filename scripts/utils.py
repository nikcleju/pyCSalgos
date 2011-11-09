# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 12:28:55 2011

@author: ncleju
"""

import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def loadshowmatrices(filename, algostr = ('GAP','SL0_approx')):
    mdict = scipy.io.loadmat(filename)
    for strname in algostr:
        print strname
        if mdict['meanmatrix'][strname][0,0].ndim == 2:
            plt.figure()
            plt.imshow(mdict['meanmatrix'][strname][0,0], cmap=cm.gray, interpolation='nearest',origin='lower')            
        elif mdict['meanmatrix'][strname][0,0].ndim == 3:
            for matrix in mdict['meanmatrix'][strname][0,0]:
                plt.figure()
                plt.imshow(matrix, cmap=cm.gray, interpolation='nearest',origin='lower')
    plt.show()
    print "Finished."
            
