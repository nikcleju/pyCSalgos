# This is a local copy of the OMP algorithm in scikit-learn,
#  to be used when scikit-learn is not available on the system
#
# Author: Vlad Niculae
# License: BSD 3 clause


"""Orthogonal matching pursuit algorithms
"""

# Author: Vlad Niculae
#
# License: BSD 3 clause

import warnings
from distutils.version import LooseVersion

import numpy as np
from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs

#from .base import LinearModel, _pre_fit
#from ..base import RegressorMixin
#from ..utils import array2d, as_float_array, check_arrays
#from ..cross_validation import _check_cv as check_cv
#from ..externals.joblib import Parallel, delayed

import scipy
solve_triangular_args = {}
if LooseVersion(scipy.__version__) >= LooseVersion('0.12'):
    solve_triangular_args = {'check_finite': False}


premature = """ Orthogonal matching pursuit ended prematurely due to linear
dependence in the dictionary. The requested precision might not have been met.
"""


def _cholesky_omp(X, y, n_nonzero_coefs, tol=None, copy_X=True,
                  return_path=False):
    """Orthogonal Matching Pursuit step using the Cholesky decomposition.

Parameters
----------
X : array, shape (n_samples, n_features)
Input dictionary. Columns are assumed to have unit norm.

y : array, shape (n_samples,)
Input targets

n_nonzero_coefs : int
Targeted number of non-zero elements

tol : float
Targeted squared error, if not None overrides n_nonzero_coefs.

copy_X : bool, optional
Whether the design matrix X must be copied by the algorithm. A false
value is only helpful if X is already Fortran-ordered, otherwise a
copy is made anyway.

return_path : bool, optional. Default: False
Whether to return every value of the nonzero coefficients along the
forward path. Useful for cross-validation.

Returns
-------
gamma : array, shape (n_nonzero_coefs,)
Non-zero elements of the solution

idx : array, shape (n_nonzero_coefs,)
Indices of the positions of the elements in gamma within the solution
vector

coef : array, shape (n_features, n_nonzero_coefs)
The first k values of column k correspond to the coefficient value
for the active features at that step. The lower left triangle contains
garbage. Only returned if ``return_path=True``.

"""
    if copy_X:
        X = X.copy('F')
    else: # even if we are allowed to overwrite, still copy it if bad order
        X = np.asfortranarray(X)

    min_float = np.finfo(X.dtype).eps
    nrm2, swap = linalg.get_blas_funcs(('nrm2', 'swap'), (X,))
    potrs, = get_lapack_funcs(('potrs',), (X,))

    alpha = np.dot(X.T, y)
    residual = y
    gamma = np.empty(0)
    n_active = 0
    indices = np.arange(X.shape[1]) # keeping track of swapping

    max_features = X.shape[1] if tol is not None else n_nonzero_coefs
    L = np.empty((max_features, max_features), dtype=X.dtype)
    L[0, 0] = 1.
    if return_path:
        coefs = np.empty_like(L)

    while True:
        lam = np.argmax(np.abs(np.dot(X.T, residual)))
        if lam < n_active or alpha[lam] ** 2 < min_float:
            # atom already selected or inner product too small
            warnings.warn(premature, RuntimeWarning, stacklevel=2)
            break
        if n_active > 0:
            # Updates the Cholesky decomposition of X' X
            L[n_active, :n_active] = np.dot(X[:, :n_active].T, X[:, lam])
            linalg.solve_triangular(L[:n_active, :n_active],
                                    L[n_active, :n_active],
                                    trans=0, lower=1,
                                    overwrite_b=True,
                                    **solve_triangular_args)
            v = nrm2(L[n_active, :n_active]) ** 2
            if 1 - v <= min_float: # selected atoms are dependent
                warnings.warn(premature, RuntimeWarning, stacklevel=2)
                break
            L[n_active, n_active] = np.sqrt(1 - v)
        X.T[n_active], X.T[lam] = swap(X.T[n_active], X.T[lam])
        alpha[n_active], alpha[lam] = alpha[lam], alpha[n_active]
        indices[n_active], indices[lam] = indices[lam], indices[n_active]
        n_active += 1
        # solves LL'x = y as a composition of two triangular systems
        gamma, _ = potrs(L[:n_active, :n_active], alpha[:n_active], lower=True,
                         overwrite_b=False)
        if return_path:
            coefs[:n_active, n_active - 1] = gamma
        residual = y - np.dot(X[:, :n_active], gamma)
        if tol is not None and nrm2(residual) ** 2 <= tol:
            break
        elif n_active == max_features:
            break

    if return_path:
        return gamma, indices[:n_active], coefs[:, :n_active]
    else:
        return gamma, indices[:n_active]


def _gram_omp(Gram, Xy, n_nonzero_coefs, tol_0=None, tol=None,
              copy_Gram=True, copy_Xy=True, return_path=False):
    """Orthogonal Matching Pursuit step on a precomputed Gram matrix.

This function uses the the Cholesky decomposition method.

Parameters
----------
Gram : array, shape (n_features, n_features)
Gram matrix of the input data matrix

Xy : array, shape (n_features,)
Input targets

n_nonzero_coefs : int
Targeted number of non-zero elements

tol_0 : float
Squared norm of y, required if tol is not None.

tol : float
Targeted squared error, if not None overrides n_nonzero_coefs.

copy_Gram : bool, optional
Whether the gram matrix must be copied by the algorithm. A false
value is only helpful if it is already Fortran-ordered, otherwise a
copy is made anyway.

copy_Xy : bool, optional
Whether the covariance vector Xy must be copied by the algorithm.
If False, it may be overwritten.

return_path : bool, optional. Default: False
Whether to return every value of the nonzero coefficients along the
forward path. Useful for cross-validation.

Returns
-------
gamma : array, shape (n_nonzero_coefs,)
Non-zero elements of the solution

idx : array, shape (n_nonzero_coefs,)
Indices of the positions of the elements in gamma within the solution
vector

coefs : array, shape (n_features, n_nonzero_coefs)
The first k values of column k correspond to the coefficient value
for the active features at that step. The lower left triangle contains
garbage. Only returned if ``return_path=True``.

"""
    Gram = Gram.copy('F') if copy_Gram else np.asfortranarray(Gram)

    if copy_Xy:
        Xy = Xy.copy()

    min_float = np.finfo(Gram.dtype).eps
    nrm2, swap = linalg.get_blas_funcs(('nrm2', 'swap'), (Gram,))
    potrs, = get_lapack_funcs(('potrs',), (Gram,))

    indices = np.arange(len(Gram)) # keeping track of swapping
    alpha = Xy
    tol_curr = tol_0
    delta = 0
    gamma = np.empty(0)
    n_active = 0

    max_features = len(Gram) if tol is not None else n_nonzero_coefs
    L = np.empty((max_features, max_features), dtype=Gram.dtype)
    L[0, 0] = 1.
    if return_path:
        coefs = np.empty_like(L)

    while True:
        lam = np.argmax(np.abs(alpha))
        if lam < n_active or alpha[lam] ** 2 < min_float:
            # selected same atom twice, or inner product too small
            warnings.warn(premature, RuntimeWarning, stacklevel=3)
            break
        if n_active > 0:
            L[n_active, :n_active] = Gram[lam, :n_active]
            linalg.solve_triangular(L[:n_active, :n_active],
                                    L[n_active, :n_active],
                                    trans=0, lower=1,
                                    overwrite_b=True,
                                    **solve_triangular_args)
            v = nrm2(L[n_active, :n_active]) ** 2
            if 1 - v <= min_float: # selected atoms are dependent
                warnings.warn(premature, RuntimeWarning, stacklevel=3)
                break
            L[n_active, n_active] = np.sqrt(1 - v)
        Gram[n_active], Gram[lam] = swap(Gram[n_active], Gram[lam])
        Gram.T[n_active], Gram.T[lam] = swap(Gram.T[n_active], Gram.T[lam])
        indices[n_active], indices[lam] = indices[lam], indices[n_active]
        Xy[n_active], Xy[lam] = Xy[lam], Xy[n_active]
        n_active += 1
        # solves LL'x = y as a composition of two triangular systems
        gamma, _ = potrs(L[:n_active, :n_active], Xy[:n_active], lower=True,
                         overwrite_b=False)
        if return_path:
            coefs[:n_active, n_active - 1] = gamma
        beta = np.dot(Gram[:, :n_active], gamma)
        alpha = Xy - beta
        if tol is not None:
            tol_curr += delta
            delta = np.inner(gamma, beta[:n_active])
            tol_curr -= delta
            if abs(tol_curr) <= tol:
                break
        elif n_active == max_features:
            break

    if return_path:
        return gamma, indices[:n_active], coefs[:, :n_active]
    else:
        return gamma, indices[:n_active]


def orthogonal_mp(X, y, n_nonzero_coefs=None, tol=None, precompute=False,
                  copy_X=True, return_path=False, precompute_gram=None):
    """Orthogonal Matching Pursuit (OMP)

Solves n_targets Orthogonal Matching Pursuit problems.
An instance of the problem has the form:

When parametrized by the number of non-zero coefficients using
`n_nonzero_coefs`:
argmin ||y - X\gamma||^2 subject to ||\gamma||_0 <= n_{nonzero coefs}

When parametrized by error using the parameter `tol`:
argmin ||\gamma||_0 subject to ||y - X\gamma||^2 <= tol

Parameters
----------
X: array, shape (n_samples, n_features)
Input data. Columns are assumed to have unit norm.

y: array, shape (n_samples,) or (n_samples, n_targets)
Input targets

n_nonzero_coefs: int
Desired number of non-zero entries in the solution. If None (by
default) this value is set to 10% of n_features.

tol: float
Maximum norm of the residual. If not None, overrides n_nonzero_coefs.

precompute: {True, False, 'auto'},
Whether to perform precomputations. Improves performance when n_targets
or n_samples is very large.

copy_X: bool, optional
Whether the design matrix X must be copied by the algorithm. A false
value is only helpful if X is already Fortran-ordered, otherwise a
copy is made anyway.

return_path: bool, optional. Default: False
Whether to return every value of the nonzero coefficients along the
forward path. Useful for cross-validation.

Returns
-------
coef: array, shape (n_features,) or (n_features, n_targets)
Coefficients of the OMP solution. If `return_path=True`, this contains
the whole coefficient path. In this case its shape is
(n_features, n_features) or (n_features, n_targets, n_features) and
iterating over the last axis yields coefficients in increasing order
of active features.

See also
--------
OrthogonalMatchingPursuit
orthogonal_mp_gram
lars_path
decomposition.sparse_encode

Notes
-----
Orthogonal matching pursuit was introduced in G. Mallat, Z. Zhang,
Matching pursuits with time-frequency dictionaries, IEEE Transactions on
Signal Processing, Vol. 41, No. 12. (December 1993), pp. 3397-3415.
(http://blanche.polytechnique.fr/~mallat/papiers/MallatPursuit93.pdf)

This implementation is based on Rubinstein, R., Zibulevsky, M. and Elad,
M., Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal
Matching Pursuit Technical Report - CS Technion, April 2008.
http://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf

"""
    if precompute_gram is not None:
        warnings.warn("precompute_gram will be removed in 0.15."
                      " Use the precompute parameter.",
                      DeprecationWarning, stacklevel=2)
        precompute = precompute_gram

    del precompute_gram

    # array2d is an utility function of scikit-learn, replaced with numpy.atleast_2d
    #X = array2d(X, order='F', copy=copy_X)
    X = np.atleast_2d(X)

    copy_X = False
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, np.newaxis]
    if y.shape[1] > 1: # subsequent targets will be affected
        copy_X = True
    if n_nonzero_coefs is None and tol is None:
        # default for n_nonzero_coefs is 0.1 * n_features
        # but at least one.
        n_nonzero_coefs = max(int(0.1 * X.shape[1]), 1)
    if tol is not None and tol < 0:
        raise ValueError("Epsilon cannot be negative")
    if tol is None and n_nonzero_coefs <= 0:
        raise ValueError("The number of atoms must be positive")
    if tol is None and n_nonzero_coefs > X.shape[1]:
        raise ValueError("The number of atoms cannot be more than the number "
                         "of features")
    if precompute == 'auto':
        precompute = X.shape[0] > X.shape[1]
    if precompute:
        G = np.dot(X.T, X)
        G = np.asfortranarray(G)
        Xy = np.dot(X.T, y)
        if tol is not None:
            norms_squared = np.sum((y ** 2), axis=0)
        else:
            norms_squared = None
        return orthogonal_mp_gram(G, Xy, n_nonzero_coefs, tol, norms_squared,
                                  copy_Gram=copy_X, copy_Xy=False)

    if return_path:
        coef = np.zeros((X.shape[1], y.shape[1], X.shape[1]))
    else:
        coef = np.zeros((X.shape[1], y.shape[1]))

    for k in range(y.shape[1]):
        out = _cholesky_omp(X, y[:, k], n_nonzero_coefs, tol,
                            copy_X=copy_X, return_path=return_path)
        if return_path:
            _, idx, coefs = out
            coef = coef[:, :, :len(idx)]
            for n_active, x in enumerate(coefs.T):
                coef[idx[:n_active + 1], k, n_active] = x[:n_active + 1]
        else:
            x, idx = out
            coef[idx, k] = x
    return np.squeeze(coef)


def orthogonal_mp_gram(Gram, Xy, n_nonzero_coefs=None, tol=None,
                       norms_squared=None, copy_Gram=True,
                       copy_Xy=True, return_path=False):
    """Gram Orthogonal Matching Pursuit (OMP)

Solves n_targets Orthogonal Matching Pursuit problems using only
the Gram matrix X.T * X and the product X.T * y.

Parameters
----------
Gram: array, shape (n_features, n_features)
Gram matrix of the input data: X.T * X

Xy: array, shape (n_features,) or (n_features, n_targets)
Input targets multiplied by X: X.T * y

n_nonzero_coefs: int
Desired number of non-zero entries in the solution. If None (by
default) this value is set to 10% of n_features.

tol: float
Maximum norm of the residual. If not None, overrides n_nonzero_coefs.

norms_squared: array-like, shape (n_targets,)
Squared L2 norms of the lines of y. Required if tol is not None.

copy_Gram: bool, optional
Whether the gram matrix must be copied by the algorithm. A false
value is only helpful if it is already Fortran-ordered, otherwise a
copy is made anyway.

copy_Xy: bool, optional
Whether the covariance vector Xy must be copied by the algorithm.
If False, it may be overwritten.

return_path: bool, optional. Default: False
Whether to return every value of the nonzero coefficients along the
forward path. Useful for cross-validation.

Returns
-------
coef: array, shape (n_features,) or (n_features, n_targets)
Coefficients of the OMP solution. If `return_path=True`, this contains
the whole coefficient path. In this case its shape is
(n_features, n_features) or (n_features, n_targets, n_features) and
iterating over the last axis yields coefficients in increasing order
of active features.

See also
--------
OrthogonalMatchingPursuit
orthogonal_mp
lars_path
decomposition.sparse_encode

Notes
-----
Orthogonal matching pursuit was introduced in G. Mallat, Z. Zhang,
Matching pursuits with time-frequency dictionaries, IEEE Transactions on
Signal Processing, Vol. 41, No. 12. (December 1993), pp. 3397-3415.
(http://blanche.polytechnique.fr/~mallat/papiers/MallatPursuit93.pdf)

This implementation is based on Rubinstein, R., Zibulevsky, M. and Elad,
M., Efficient Implementation of the K-SVD Algorithm using Batch Orthogonal
Matching Pursuit Technical Report - CS Technion, April 2008.
http://www.cs.technion.ac.il/~ronrubin/Publications/KSVD-OMP-v2.pdf

"""
    # array2d is an utility function of scikit-learn, replaced with numpy.atleast_2d
    #Gram = array2d(Gram, order='F', copy=copy_Gram)
    Gram = np.atleast_2d(Gram)

    Xy = np.asarray(Xy)
    if Xy.ndim > 1 and Xy.shape[1] > 1:
        # or subsequent target will be affected
        copy_Gram = True
    if Xy.ndim == 1:
        Xy = Xy[:, np.newaxis]
        if tol is not None:
            norms_squared = [norms_squared]

    if n_nonzero_coefs is None and tol is None:
        n_nonzero_coefs = int(0.1 * len(Gram))
    if tol is not None and norms_squared is None:
        raise ValueError('Gram OMP needs the precomputed norms in order '
                         'to evaluate the error sum of squares.')
    if tol is not None and tol < 0:
        raise ValueError("Epsilon cannot be negative")
    if tol is None and n_nonzero_coefs <= 0:
        raise ValueError("The number of atoms must be positive")
    if tol is None and n_nonzero_coefs > len(Gram):
        raise ValueError("The number of atoms cannot be more than the number "
                         "of features")

    if return_path:
        coef = np.zeros((len(Gram), Xy.shape[1], len(Gram)))
    else:
        coef = np.zeros((len(Gram), Xy.shape[1]))

    for k in range(Xy.shape[1]):
        out = _gram_omp(Gram, Xy[:, k], n_nonzero_coefs,
                        norms_squared[k] if tol is not None else None, tol,
                        copy_Gram=copy_Gram, copy_Xy=copy_Xy,
                        return_path=return_path)
        if return_path:
            _, idx, coefs = out
            coef = coef[:, :, :len(idx)]
            for n_active, x in enumerate(coefs.T):
                coef[idx[:n_active + 1], k, n_active] = x[:n_active + 1]
        else:
            x, idx = out
            coef[idx, k] = x

    return np.squeeze(coef)
