# -*- coding: utf-8 -*-
"""
Created on Wed Nov 09 12:28:55 2011

@author: ncleju
"""

import numpy
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Sample call
#utils.loadshowmatrices_multipixels('H:\\CS\\Python\\Results\\pt_std1\\approx_pt_std1.mat', dosave=True, saveplotbase='approx_pt_std1_',saveplotexts=('png','eps','pdf'))

#def loadshowmatrices(filename, algonames = None):
#    mdict = scipy.io.loadmat(filename)
#    if algonames == None:
#      algonames = mdict['algonames']
#    
#    for algonameobj in algonames:
#        algoname = algonameobj[0][0]
#        print algoname
#        if mdict['meanmatrix'][algoname][0,0].ndim == 2:
#            plt.figure()
#            plt.imshow(mdict['meanmatrix'][algoname][0,0], cmap=cm.gray, interpolation='nearest',origin='lower')            
#        elif mdict['meanmatrix'][algoname][0,0].ndim == 3:
#            for matrix in mdict['meanmatrix'][algoname][0,0]:
#                plt.figure()
#                plt.imshow(matrix, cmap=cm.gray, interpolation='nearest',origin='lower')
#    plt.show()
#    print "Finished."
#
#
#def loadsavematrices(filename, saveplotbase, saveplotexts, algonames = None):
#    
#    mdict = scipy.io.loadmat(filename)
#    lambdas = mdict['lambdas']
#
#    if algonames is None:
#      algonames = mdict['algonames']
#    
#    for algonameobj in algonames:
#        algoname = algonameobj[0][0]
#        print algoname
#        if mdict['meanmatrix'][algoname][0,0].ndim == 2:
#            plt.figure()
#            plt.imshow(mdict['meanmatrix'][algoname][0,0], cmap=cm.gray, interpolation='nearest',origin='lower')
#            for ext in saveplotexts:
#              plt.savefig(saveplotbase + algoname + '.' + ext, bbox_inches='tight')
#        elif mdict['meanmatrix'][algoname][0,0].ndim == 3:
#            ilbd = 0
#            for matrix in mdict['meanmatrix'][algoname][0,0]:
#                plt.figure()
#                plt.imshow(matrix, cmap=cm.gray, interpolation='nearest',origin='lower')
#                for ext in saveplotexts:
#                  plt.savefig(saveplotbase + algoname + ('_lbd%.0e' % lambdas[ilbd]) + '.' + ext, bbox_inches='tight')
#                ilbd = ilbd + 1
#    print "Finished."  
    
def loadmatrices(filename, algonames=None, doshow=True, dosave=False, saveplotbase=None, saveplotexts=None):
    
    if dosave and (saveplotbase is None or saveplotexts is None):
      print('Error: please specify name and extensions for saving')
      raise Exception('Name or extensions for saving not specified')
    
    mdict = scipy.io.loadmat(filename)

    if dosave:
      lambdas = mdict['lambdas']

    if algonames is None:
      algonames = mdict['algonames']
    
    for algonameobj in algonames:
        algoname = algonameobj[0][0]
        print algoname
        if mdict['meanmatrix'][algoname][0,0].ndim == 2:
            plt.figure()
            plt.imshow(mdict['meanmatrix'][algoname][0,0], cmap=cm.gray, interpolation='nearest',origin='lower')
            if dosave:
              for ext in saveplotexts:
                plt.savefig(saveplotbase + algoname + '.' + ext, bbox_inches='tight')
        elif mdict['meanmatrix'][algoname][0,0].ndim == 3:
            if dosave:
              ilbd = 0
            for matrix in mdict['meanmatrix'][algoname][0,0]:
                plt.figure()
                plt.imshow(matrix, cmap=cm.gray, interpolation='nearest',origin='lower')
                if dosave:
                  for ext in saveplotexts:
                    plt.savefig(saveplotbase + algoname + ('_lbd%.0e' % lambdas[ilbd]) + '.' + ext, bbox_inches='tight')
                  ilbd = ilbd + 1
    
    if doshow:
      plt.show()
    print "Finished."    

    
def loadshowmatrices_multipixels(filename, algonames = None, doshow=True, dosave=False, saveplotbase=None, saveplotexts=None):
  
    if dosave and (saveplotbase is None or saveplotexts is None):
      print('Error: please specify name and extensions for saving')
      raise Exception('Name or extensions for saving not specified')
      
    mdict = scipy.io.loadmat(filename)
    
    if dosave:
      lambdas = mdict['lambdas']
      
    if algonames == None:
      algonames = mdict['algonames']
    
#    thresh = 0.90
    N = 10
#    border = 2
#    bordercolor = 0 # black
    
    for algonameobj in algonames:
        algoname = algonameobj[0][0]
        print algoname
        if mdict['meanmatrix'][algoname][0,0].ndim == 2:
            
            # Prepare bigger matrix
            rows,cols = mdict['meanmatrix'][algoname][0,0].shape
            bigmatrix = numpy.zeros((N*rows,N*cols))
            for i in numpy.arange(rows):
              for j in numpy.arange(cols):
                bigmatrix[i*N:i*N+N,j*N:j*N+N] = mdict['meanmatrix'][algoname][0,0][i,j]
            bigmatrix = int_drawseparation(mdict['meanmatrix'][algoname][0,0],bigmatrix,10,0.9,2,0)
            bigmatrix = int_drawseparation(mdict['meanmatrix'][algoname][0,0],bigmatrix,10,0.8,2,0.5)
#                # Mark 95% border
#                if mdict['meanmatrix'][algoname][0,0][i,j] > thresh:
#                  # Top border
#                  if mdict['meanmatrix'][algoname][0,0][i-1,j] < thresh and i>0:
#                    bigmatrix[i*N:i*N+border,j*N:j*N+N] = bordercolor
#                  # Bottom border
#                  if mdict['meanmatrix'][algoname][0,0][i+1,j] < thresh and i<rows-1:
#                    bigmatrix[i*N+N-border:i*N+N,j*N:j*N+N] = bordercolor                
#                  # Left border
#                  if mdict['meanmatrix'][algoname][0,0][i,j-1] < thresh and j>0:
#                    bigmatrix[i*N:i*N+N,j*N:j*N+border] = bordercolor
#                  # Right border (not very probable)
#                  if j<cols-1 and mdict['meanmatrix'][algoname][0,0][i,j+1] < thresh:
#                    bigmatrix[i*N:i*N+N,j*N+N-border:j*N+N] = bordercolor
                    
            plt.figure()
            #plt.imshow(mdict['meanmatrix'][algoname][0,0], cmap=cm.gray, interpolation='nearest',origin='lower')            
            plt.imshow(bigmatrix, cmap=cm.gray, interpolation='nearest',origin='lower')        
            if dosave:
              for ext in saveplotexts:
                plt.savefig(saveplotbase + algoname + '.' + ext, bbox_inches='tight')            
        elif mdict['meanmatrix'][algoname][0,0].ndim == 3:
            if dosave:
              ilbd = 0          
              
            for matrix in mdict['meanmatrix'][algoname][0,0]:
              
                # Prepare bigger matrix
                rows,cols = matrix.shape
                bigmatrix = numpy.zeros((N*rows,N*cols))
                for i in numpy.arange(rows):
                  for j in numpy.arange(cols):
                    bigmatrix[i*N:i*N+N,j*N:j*N+N] = matrix[i,j]
                bigmatrix = int_drawseparation(matrix,bigmatrix,10,0.9,2,0)
                bigmatrix = int_drawseparation(matrix,bigmatrix,10,0.8,2,0.5)
#                    # Mark 95% border
#                    if matrix[i,j] > thresh:
#                      # Top border
#                      if matrix[i-1,j] < thresh and i>0:
#                        bigmatrix[i*N:i*N+border,j*N:j*N+N] = bordercolor
#                      # Bottom border
#                      if matrix[i+1,j] < thresh and i<rows-1:
#                        bigmatrix[i*N+N-border:i*N+N,j*N:j*N+N] = bordercolor                
#                      # Left border
#                      if matrix[i,j-1] < thresh and j>0:
#                        bigmatrix[i*N:i*N+N,j*N:j*N+border] = bordercolor
#                      # Right border (not very probable)
#                      if j<cols-1 and matrix[i,j+1] < thresh:
#                        bigmatrix[i*N:i*N+N,j*N+N-border:j*N+N] = bordercolor
                
                plt.figure()
                #plt.imshow(matrix, cmap=cm.gray, interpolation='nearest',origin='lower')
                plt.imshow(bigmatrix, cmap=cm.gray, interpolation='nearest',origin='lower')
                if dosave:
                  for ext in saveplotexts:
                    plt.savefig(saveplotbase + algoname + ('_lbd%.0e' % lambdas[ilbd]) + '.' + ext, bbox_inches='tight')
                  ilbd = ilbd + 1                
    if doshow:
      plt.show()
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
  
def int_drawseparation(matrix,bigmatrix,N,thresh,border,bordercolor):
  rows,cols = matrix.shape
  for i in numpy.arange(rows):
    for j in numpy.arange(cols):
      # Mark border
      # Use top-left corner of current square for reference
      if matrix[i,j] > thresh:
        # Top border
        if matrix[i-1,j] < thresh and i>0:
          bigmatrix[i*N:i*N+border,j*N:j*N+N] = bordercolor
        # Bottom border
        if i<rows-1 and matrix[i+1,j] < thresh:
          bigmatrix[i*N+N-border:i*N+N,j*N:j*N+N] = bordercolor                
        # Left border
        if matrix[i,j-1] < thresh and j>0:
          bigmatrix[i*N:i*N+N,j*N:j*N+border] = bordercolor
        # Right border (not very probable)
        if j<cols-1 and matrix[i,j+1] < thresh:
          bigmatrix[i*N:i*N+N,j*N+N-border:j*N+N] = bordercolor  
  
  return bigmatrix