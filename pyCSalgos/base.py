"""
Defines abstract base classes for solvers: 
 - Class SparseSolver for sparse coding algorithms
 - Class AnalysisSparseSolver for analysis-based recovery problems
"""

# Author: Nicolae Cleju
# License: BSD 3 clause


from six import with_metaclass
from sklearn.base import BaseEstimator
from abc import ABCMeta, abstractmethod


# TODO: Could be derived from LinearModel instead?
class SparseSolver(with_metaclass(ABCMeta, BaseEstimator)):
    """
    Base class for all synthesis-based solvers.

    All synthesis solvers are derived from this class, and implement a solve() method that performs the actual solving.

    Tips for writing your own solver:
    - Derive from SparseSolver (or AnalysisSparseSolver if appropriate).
    - All parameters should be set in the derived class' __init__(). A solver object should hold all
      the parameters required for solving, but not the actual data or the results.
    - Implement solve() method. This takes the data and provides the result. Nothing is stored in the solver object.

    Other notes (from scikit-learn):
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their __init__ as explicit keyword
    arguments (no *args, **kwargs).    
    """

    #__metaclass__ = ABCMeta
    
    @abstractmethod
    def solve(self, data, dictionary, realdict=None):
        """
        Performs the solving

        :param data: The data vector or matrix. If a matrix, it should contain multiple vectors as columns
        :param dictionary: The dictionary, with columnwise atoms

        :return: The coefficient matrix. Each column is the decomposition of the corresponding data vector
         """

    # __repr()__ mist return a string object, not an unicode object!
    #def __repr__(self):
    #    """
    #    Unicode version of __repr__()
    #    """
    #    return unicode(super(SparseSolver,self).__repr__())

class AnalysisSparseSolver(with_metaclass(ABCMeta, BaseEstimator)):
    """
    Base class for all analysis-based solvers

    All analysis solvers are derived from this class, and implement a solve() method that performs the actual solving.

    Tips for writing your own solver:
    - Derive from AnalysisSparseSolver.
    - All parameters should be set in the derived class' __init__(). A solver object should hold all
      the parameters required for solving, but not the actual data or the results.
    - Implement solve() method. This takes the data and provides the result. Nothing is stored in the solver object.


    Notes (from scikit-learn):
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their __init__ as explicit keyword
    arguments (no *args, **kwargs).
    """

    #__metaclass__ = ABCMeta

    @abstractmethod
    def solve(self, measurements, acqumatrix, operator, realdict):
        """
        Performs the solving

        :param measurements: The measurements vector or matrix. If a matrix, it should contain multiple vectors
         as columns
        :param acqumatrix: The acquisition matrix
        :param operator: The operator matrix
        :param realdict:

        :return: The recovered signals. Each column is the signal recovered from the corresponding measurements vector
         """

class ERCcheckMixin:
    """
    Provides a base for Exact-Recovery-Check method for solvers.
    Use this mixin as an additional base class in solvers, either synthesis or analysis,
     that have an ERC condition that can be checked.
    """

    def checkERC(self, acqumatrix, dictoper, support):
        """
        Checks Exact Recovery Condition (ERC), if any, of this solver for given support

        :param acqumatrix: The acquisition matrix
        :param operator: The dictionry OR operator matrix (typically fat or tall matrix, respectively)
        :param support: A matrix containing, as columns, the support (or cosupport, in case of analysis)
         for each signal to check

        :return: A vector of True/False containing the result of ERC check
         """
        RuntimeError("Solver "+ self +" does not have an ERC condition implemented")

