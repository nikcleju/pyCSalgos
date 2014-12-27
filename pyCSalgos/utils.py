"""
utils.py

Utility functions
"""
__author__ = 'ncleju'

import numpy as np
import scipy

# Only returns the first parameter of scipy.linalg.lstsq()
def fast_lstsq(A, y):
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
