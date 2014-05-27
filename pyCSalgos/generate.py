"""
Contains function to generate datasets and problems
"""

# Author: Nicolae Cleju
# License: BSD 3 clause

import numpy

try:
    import sklearn.datasets
    has_sklearn_datasets = True
except ImportError, e:
    # module doesn't exist
    has_sklearn_datasets = False

def make_sparse_coded_signal(n,N,k,Ndata,use_sklearn=True):
    """
    Generate a sparse coded signal
    """
    if use_sklearn and has_sklearn_datasets:
        return sklearn.datasets.make_sparse_coded_signal(n_samples=Ndata, n_features=n, n_components=N, n_nonzero_coefs=k)

    # Generate random dictionary and normalize
    # TODO: Use sklearn-type random generators instead of numpy.random
    D = numpy.random.randn(n,N)
    D = D / numpy.sqrt(numpy.sum(D**2, axis=0))

    # Generate coefficients matrix
    support = numpy.zeros((k, Ndata),dtype=int)
    gamma = numpy.zeros((N, Ndata))
    for i in range(Ndata):
        support[:, i] = numpy.random.permutation(N)[:k]
        gamma[support[:, i],i] = numpy.random.randn(k)


    # Generate data
    X = numpy.dot(D,gamma)

    return X,D,gamma,support



