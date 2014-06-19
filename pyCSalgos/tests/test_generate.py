"""
Tests for generate.py
"""

# Author: Nicolae Cleju
# License: BSD 3 clause

import numpy
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal

from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose
from sklearn.utils.testing import assert_warns

from generate import make_sparse_coded_signal
from generate import make_compressed_sensing_problem


def test_make_sparse_coded_signal():
    """ Tests make_sparse_coded_signal()"""

    n, N = 20, 30
    k = 5
    Ndata = 10
    # Parameterized test:
    for use_sklearn in [True,False]:
        yield subtest_make_sparse_coded_signal, n, N, k, Ndata, use_sklearn

def subtest_make_sparse_coded_signal(n,N,k,Ndata,use_sklearn):

    X, D, gamma, support = make_sparse_coded_signal(n, N, k, Ndata, "randn", use_sklearn)

    # check shapes
    print X.shape
    print (n, N)

    assert_equal(X.shape, (n, Ndata), "X shape mismatch")
    assert_equal(D.shape, (n, N), "D shape mismatch")
    assert_equal(gamma.shape, (N, Ndata), "gamma shape mismatch")
    assert_equal(support.shape, (k, Ndata), "support shape mismatch")

   # check multiplication
    assert_array_equal(X, numpy.dot(D, gamma))

    # check dictionary normalization
    assert_array_almost_equal(numpy.sqrt((D ** 2).sum(axis=0)),
                              numpy.ones(D.shape[1]))

    for i in range(Ndata):
        assert(numpy.all(gamma[support[:, i], i])) # check if all support is non-zero
        izero = numpy.setdiff1d(range(N), support[:, i])
        assert(not numpy.any(gamma[izero, i])) # check if all zeros are zero


def test_make_sparse_coded_signal_dictionary():
    n, N = 20, 30
    k = 5
    Ndata = 10

    assert_raises(ValueError, make_sparse_coded_signal, n, N, k, Ndata, "orthonormal", True)

    X, D, gamma, support = make_sparse_coded_signal(n, n, k, Ndata, dictionary="orthonormal", use_sklearn=True)
    assert_allclose(numpy.dot(D, D.T), numpy.eye(n), atol=1e-10)
    assert_allclose(numpy.dot(D.T, D), numpy.eye(n), atol=1e-10)

    Dict = numpy.random.randn(n,N)
    assert_raises(ValueError, make_sparse_coded_signal, n, N+1, k, Ndata, Dict, True)
    X, D, gamma, support = make_sparse_coded_signal(n, N, k, Ndata, dictionary=Dict, use_sklearn=True)
    assert_array_equal(D, Dict)

    assert_raises(ValueError, make_sparse_coded_signal, n, N, k, Ndata, "somethingwrong", True)

def test_make_compressed_sensing_problem():
    m = 10
    n, N = 20, 30
    k = 5
    Ndata = 10

    assert_raises(ValueError, make_compressed_sensing_problem, m, n, N, k, Ndata, "randn", "somethingwrong", True)
    P = numpy.random.randn(m,n)
    assert_raises(ValueError, make_compressed_sensing_problem, m+1, n, N, k, Ndata, "randn", P, True)

    measurements, acqumatrix, data, dictionary, gamma, support = \
        make_compressed_sensing_problem(m, n, N, k, Ndata, "randn", "randn", True)

    assert_equal(measurements.shape, (m,Ndata))
    assert_equal(acqumatrix.shape, (m,n))
    assert_array_equal(measurements, numpy.dot(acqumatrix, data))
