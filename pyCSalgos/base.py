"""
Defines abstract SparseSolver base class for solvers 
"""

# Author: Nicolae Cleju
# License: BSD 3 clause


from six import with_metaclass
from  sklearn.base import BaseEstimator
from abc import ABCMeta, abstractmethod

# TODO: Could be derived from LinearModel instead?
class SparseSolver(with_metaclass(ABCMeta, BaseEstimator)):
    """
    Base class for all solvers
    
    Notes (from scikit-learn):
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their __init__ as explicit keyword
    arguments (no *args, **kwargs).    
    """

    __metaclass__ = ABCMeta
    
#    def __init__(self,y=None,A=None):
#        """Constructs a sparse solver"""
#        self.y_ = y
#        self.A_ = A
#        self.coef_ = None

    @abstractmethod
    def solve(self, data, dictionary):
        """Run the solver"""


class AnalysisSparseSolver(with_metaclass(ABCMeta, BaseEstimator)):
    """
    Base class for all solvers

    Notes (from scikit-learn):
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their __init__ as explicit keyword
    arguments (no *args, **kwargs).
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def solve(self, measurements, acqumatrix, operator):
        """Run the solver"""
