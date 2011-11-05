import numpy as np
import numpy.linalg
import scipy.io
import scipy.linalg
import time

def main():
	mdict = scipy.io.loadmat('testLstsq.mat')
	A = mdict['A'].newbyteorder('=')
	b = mdict['b']
	
	starttime = time.time()
	for i in np.arange(mdict['nruns']):
	#for i in np.arange(100):
		#np.linalg.lstsq(A, b)
		np.linalg.svd(A)
	print "Elapsed time = ",(time.time() - starttime)

if __name__ == "__main__":
	main()
