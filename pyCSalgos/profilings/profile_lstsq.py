# Run all this in ipython

import numpy as np
import scipy.linalg

def fast_lsqst(A, y):
    m,n = A.shape
    if m >= n:
        Q, R = scipy.linalg.qr(A, mode='economic') # qr decomposition of A
        Qb = np.dot(Q.T,y) # computing Q^T*b (project b onto the range of A)
        x_qr = scipy.linalg.solve_triangular(R, Qb)
    else:
        # http://en.wikipedia.org/wiki/QR_decomposition#Using_for_solution_to_linear_inverse_problems
        Q, R = scipy.linalg.qr(A.T, mode='economic') # qr decomposition of A.T
        x_qr = np.dot(Q, scipy.linalg.solve_triangular(R, y))
    return x_qr

N = 200
n = 240
A = np.random.randn(N, n)
y = np.random.randn(N)
%timeit scipy.linalg.lstsq(A, y)
%timeit np.linalg.lstsq(A, y)
%timeit np.dot(scipy.linalg.pinv(A), y)
%timeit np.dot(np.linalg.pinv(A), y)
%timeit fast_lsqst(A,y)

# Seems like QR decomposition cu solve_triangular (last one) is best!

