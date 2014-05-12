# -*- coding: utf-8 -*-
"""
Tests for base.py
"""

# Heavily inspired from test_base.py in scikit-learn

# Author: Nicolae Cleju
# License: BSD 3 clause


import numpy as np

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_false
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_not_equal
from sklearn.utils.testing import assert_raises

from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_less
import numpy as np

import warnings
from sklearn.utils import deprecated

import pyCSalgos
from pyCSalgos.base import SparseSolver


#############################################################################
# Test classes

class MySolver(SparseSolver):

    def __init__(self, y=None, A=None):
        self.y_ = y
        self.A_ = A
        self.coef_ = None        
    def run(self):
        """Do nothing"""


class K(SparseSolver):
    def __init__(self, c=None, d=None):
        self.c = c
        self.d = d
    def run(self):
        """Do nothing"""


class T(SparseSolver):
    def __init__(self, a=None, b=None):
        self.a = a
        self.b = b
    def run(self):
        """Do nothing"""


class DeprecatedAttributeSolver(SparseSolver):
    def __init__(self, a=None, b=None):
        self._a = a
        if b is not None:
            #DeprecationWarning("b is deprecated and renamed 'a'")
            warnings.warn("b is deprecated and renamed 'a'", DeprecationWarning)
            self._a = b
   
    def run(self):
        """Do nothing"""

    @property
    @deprecated("Parameter 'b' is deprecated and renamed to 'a'")
    def b(self):
        return self._a


#############################################################################
# Tests

def test_instantiation():
    """ Should not allow instantiation of abstract classes"""
    assert_raises(TypeError,SparseSolver)

def test_repr():
    """Smoke test the repr of the base estimator."""
    my_solver = MySolver()
    repr(my_solver)
    test = T(K(), K())
    assert_equal(
        repr(test),
        "T(a=K(c=None, d=None), b=K(c=None, d=None))"
    )

    some_est = T(a=["long_params"] * 1000)
    assert_equal(len(repr(some_est)), 415)


def test_str():
    """Smoke test the str of the base estimator"""
    my_solver = MySolver()
    str(my_solver)


def test_get_params():
    test = T(K(), K())

    assert_true('a__d' in test.get_params(deep=True))
    assert_true('a__d' not in test.get_params(deep=False))

    test.set_params(a__d=2)
    assert_true(test.a.d == 2)
    assert_raises(ValueError, test.set_params, a__a=2)

def test_deprecated():
    """Test if deprecation warning is issued"""
    
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        d = DeprecatedAttributeSolver(a=1,b=2)
        d.run()
        print len(w)
        # Verify some things
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "deprecated" in str(w[-1].message)    

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        x=d.b
        print len(w)
        # Verify some things
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "deprecated" in str(w[-1].message)    


def test_get_params_deprecated():
    # deprecated attribute should not show up as params
    est = DeprecatedAttributeSolver(a=1)

    assert_true('a' in est.get_params())
    assert_true('a' in est.get_params(deep=True))
    assert_true('a' in est.get_params(deep=False))

    assert_true('b' not in est.get_params())
    assert_true('b' not in est.get_params(deep=True))
    assert_true('b' not in est.get_params(deep=False))