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


def test_make_sparse_coded_signal():
    """ Tests make_sparse_coded_signal()"""

    n, N = 20, 30
    k = 5
    Ndata = 10
    # Parameterized test:
    for use_sklearn in [True,False]:
        yield subtest_make_sparse_coded_signal, n, N, k, Ndata, use_sklearn

def subtest_make_sparse_coded_signal(n,N,k,Ndata,use_sklearn):

    X, D, gamma, support = make_sparse_coded_signal(n, N, k, Ndata, use_sklearn)

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


def test_dictionary():
    n, N = 20, 30
    k = 5
    Ndata = 10

    assert_raises(ValueError, make_sparse_coded_signal, n, N, k, Ndata, True, "orthonormal")

    X, D, gamma, support = make_sparse_coded_signal(n, n, k, Ndata, use_sklearn=True, dictionary="orthonormal")
    assert_allclose(numpy.dot(D, D.T), numpy.eye(n), atol=1e-10)
    assert_allclose(numpy.dot(D.T, D), numpy.eye(n), atol=1e-10)

    Dict = numpy.random.randn(n,N)
    assert_raises(ValueError, make_sparse_coded_signal, n, N+1, k, Ndata, True, Dict)
    X, D, gamma, support = make_sparse_coded_signal(n, N, k, Ndata, use_sklearn=True, dictionary=Dict)
    assert_array_equal(D, Dict)

    assert_raises(ValueError, make_sparse_coded_signal, n, N, k, Ndata, True, "somethingwrong")
