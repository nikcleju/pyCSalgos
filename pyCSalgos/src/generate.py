"""
Contains function to generate datasets and problems
"""

# Author: Nicolae Cleju
# License: BSD 3 clause

import numpy
import scipy

try:
    import sklearn.datasets
    has_sklearn_datasets = True
except ImportError, e:
    # module doesn't exist
    has_sklearn_datasets = False

def make_sparse_coded_signal(n,N,k,Ndata,use_sklearn=True, dictionary="randn"):
    """
    Generate a sparse coded signal
    """

    # Generate coefficients matrix
    support = numpy.zeros((k, Ndata),dtype=int)

    if dictionary == "randn" and use_sklearn and has_sklearn_datasets:
        # use random normalized dictionary from scikit-learn
        X, D, gamma = sklearn.datasets.make_sparse_coded_signal(n_samples=Ndata, n_features=n, n_components=N, n_nonzero_coefs=k)
        for i in range(Ndata):
            support[:, i] = numpy.nonzero(gamma[:,i])[0]

    else:
        # Create dictionary
        if dictionary == "randn":
            # generate random dictionary and normalize
            # TODO: Use sklearn-type random generators instead of numpy.random
            D = numpy.random.randn(n,N)
            D = D / numpy.sqrt(numpy.sum(D**2, axis=0))
        elif dictionary == "orthonormal":
            if n != N:
                raise ValueError("Orthonormal dictionary has n==N")
            # generate random square dictionary and orthonormalize
            # TODO: Use sklearn-type random generators instead of numpy.random
            D = numpy.random.randn(n,N)
            D = scipy.linalg.orth(D)
        elif isinstance(dictionary, numpy.ndarray):
            # dictionary is given
            if n!= dictionary.shape[0] or N != dictionary.shape[1]:
                raise ValueError("Dictionary shape different from (n,N)")
            D = dictionary
        else:
            raise ValueError("Wrong dictionary parameter")

        # Generate coefficients matrix
        gamma = numpy.zeros((N, Ndata))
        for i in range(Ndata):
            support[:, i] = numpy.random.permutation(N)[:k]
            gamma[support[:, i],i] = numpy.random.randn(k)

        # Generate data
        X = numpy.dot(D,gamma)

    return X,D,gamma,support




