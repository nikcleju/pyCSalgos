"""
test_omp.py

Testing functions for OMP

Heavily inspired from test_omp in scikit-learn, copyrighted Vlad Niculae
"""

# Author: Nicolae Cleju, Vlad Niculae
# License: BSD 3 clause

# TODO: test underlying function directly instead of class?

import numpy as np
import scipy

from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose
from sklearn.utils.testing import assert_warns


from generate import make_sparse_coded_signal
from omp import OrthogonalMatchingPursuit

n, N, k, Ndata = 20,30,3,10
rng = np.random.RandomState(47)

SolverClass = OrthogonalMatchingPursuit

X, D, gamma, support = make_sparse_coded_signal(n, N, k, Ndata)
tol = 1e-6

algorithms = ["sklearn", "sklearn_local", "sparsify_QR", "sturm_QR"]

def test_correct_shapes():
    stopvals = [1e-6, k]
    for algo in algorithms:
        for stopval in stopvals:
            yield subtest_correct_shapes, stopval, algo

def subtest_correct_shapes(stopval, algorithm):
    omp = SolverClass(stopval = stopval, algorithm=algorithm)
    # single vector
    coef = omp.solve(X[:,0], D)
    assert_equal(coef.shape, (N,))
    # multiple vectors
    coef = omp.solve(X, D)
    assert_equal(coef.shape, (N, Ndata))


def test_tol():
    stopvals = [1e-6]
    for algo in algorithms:
        for stopval in stopvals:
            yield subtest_tol, stopval, algo

def subtest_tol(stopval, algorithm):
    omp = SolverClass(stopval = stopval, algorithm=algorithm)
    coef = omp.solve(X, D)
    for i in range(X.shape[1]):
        assert_true(np.sum((X[:, i] - np.dot(D, coef[:,i])) ** 2) <= stopval)


def test_n_nonzero_coefs():
    stopvals = [k]
    for algo in algorithms:
        for stopval in stopvals:
            yield subtest_n_nonzero_coefs, stopval, algo

def subtest_n_nonzero_coefs(stopval, algorithm):
    omp = SolverClass(stopval = stopval, algorithm=algorithm)
    coef = omp.solve(X, D)
    for i in range(X.shape[1]):
        assert_true(np.count_nonzero(coef[:,i]) <= stopval)

def test_perfect_support_recovery():
    stopvals = [k]
    for algo in algorithms:
        for stopval in stopvals:
            yield subtest_perfect_support_recovery, stopval, algo

def subtest_perfect_support_recovery(stopval, algorithm):
    # check support only when stopping criterion = fixed sparsity
    # otherwise might get very small but non-zero coefficients
    omp = SolverClass(stopval = stopval, algorithm=algorithm)
    notused, Dortho, gammaortho, supportortho = make_sparse_coded_signal(n, n, k, Ndata)
    Dortho = scipy.linalg.orth(Dortho)
    Xortho = np.dot(Dortho, gammaortho)
    coef = omp.solve(Xortho, Dortho)
    for i in range(Xortho.shape[1]):
        assert_array_equal(supportortho[:,i], np.flatnonzero(coef[:,i]))   # check support
    #assert_array_almost_equal(gamma[:, i], coef[:,i], decimal=2)
    assert_allclose(gammaortho, coef, atol=1e-10)


def test_perfect_signal_recovery():
    stopvals = [1e-6]
    for algo in algorithms:
        for stopval in stopvals:
            yield subtest_perfect_signal_recovery, stopval, algo

def subtest_perfect_signal_recovery(stopval, algorithm):
    omp = SolverClass(stopval = stopval, algorithm=algorithm)
    coef = omp.solve(X,D)
    assert_allclose(gamma, coef, atol=1e-6)


def test_omp_reaches_least_squares():
    for algo in algorithms:
        yield subtest_omp_reaches_least_squares, algo

def subtest_omp_reaches_least_squares(algorithm):
    n1 = 10
    N1 = 8
    X1 = rng.randn(n1, 3)
    D1 = rng.randn(n1, N1)
    for i in range(N1):
        D1[:,i] = D1[:,i] / np.linalg.norm(D1[:,i])
    omp = SolverClass(stopval = N1, algorithm=algorithm)
    coef = omp.solve(X1, D1)
    lstsq = np.dot(np.linalg.pinv(D1), X1)
    assert_allclose(coef, lstsq, atol=1e-10)


def test_bad_input():
    assert_raises(ValueError, SolverClass, stopval=-1)

    omp = SolverClass(stopval=n+1, algorithm="sklearn")
    assert_raises(ValueError, omp.solve, X, D)

    omp = SolverClass(stopval=N+1, algorithm="sklearn")
    assert_raises(ValueError, omp.solve, X, D)

    omp = SolverClass(stopval=1e-6, algorithm="nonexistent")
    assert_raises(ValueError, omp.solve, X, D)


# TODO: to reanalyze, didn't figure out what it does
def test_identical_regressors():
    newD = D.copy()
    newD[:, 1] = newD[:, 0]
    gamma = np.zeros(N)
    gamma[0] = gamma[1] = 1.
    newy = np.dot(newD, gamma)
    omp = SolverClass(stopval=2, algorithm="sklearn")
    assert_warns(RuntimeWarning, omp.solve, data=newy, dictionary=newD)
    #assert_warns(RuntimeWarning, orthogonal_mp, newX, newy, 2)


# TODO: to reanalyze, didn't figure out what it does
def test_swapped_regressors():
    gamma = np.zeros(N)
    # X[:, 21] should be selected first, then X[:, 0] selected second,
    # which will take X[:, 21]'s place in case the algorithm does
    # column swapping for optimization (which is the case at the moment)
    gamma[21] = 1.0
    gamma[0] = 0.5
    new_y = np.dot(D, gamma)
    new_Xy = np.dot(D.T, new_y)
    #gamma_hat = orthogonal_mp(X, new_y, 2)
    gamma_hat = SolverClass(stopval=2).solve(new_y, D)
    assert_array_equal(np.flatnonzero(gamma_hat), [0, 21])


# TODO instead: check same results with QR and localSKlearn
# def test_with_without_gram():
#     assert_array_almost_equal(
#         orthogonal_mp(X, y, n_nonzero_coefs=5),
#         orthogonal_mp(X, y, n_nonzero_coefs=5, precompute=True))
#
#
# def test_with_without_gram_tol():
#     assert_array_almost_equal(
#         orthogonal_mp(X, y, tol=1.),
#         orthogonal_mp(X, y, tol=1., precompute=True))
#
#

# NOT SURE
# def test_unreachable_accuracy():
#     assert_array_almost_equal(
#         orthogonal_mp(X, y, tol=0),
#         orthogonal_mp(X, y, n_nonzero_coefs=n_features))
#
#     assert_array_almost_equal(
#         assert_warns(RuntimeWarning, orthogonal_mp, X, y, tol=0,
#                      precompute=True),
#         orthogonal_mp(X, y, precompute=True,
#                       n_nonzero_coefs=n_features))
#
#



