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

def make_sparse_coded_signal(n,N,k,Ndata, dictionary="randn", use_sklearn=True):
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


def make_compressed_sensing_problem(m, n,N,k,Ndata, dictionary="randn", acquisition="randn", use_sklearn=True):
    """
    Make compressed sensing problem
    """

    # generate sparse coded data
    data, dictionary, gamma, support = make_sparse_coded_signal(n,N,k,Ndata, dictionary, use_sklearn)

    # generate acquisition matrix
    if acquisition=="randn":
        acqumatrix = numpy.random.randn(m, n)
    elif isinstance(acquisition, numpy.ndarray):
        # acquisition matrix is given
        if m != acquisition.shape[0] or n != acquisition.shape[1]:
            raise ValueError("Acquisition matrix shape different from (m,n)")
        acqumatrix = acquisition
    else:
        raise ValueError("Unrecognized acquisition matrix type")

    measurements = numpy.dot(acqumatrix, data)

    return measurements, acqumatrix, data, dictionary, gamma, support



def make_cosparse_coded_signal(n, N, l, numdata, operatortype="tightframe"):
    """
    Generate co-sparse coded signals
    """

    # Prepare matrices
    data = numpy.zeros((n, numdata))
    cosupport = numpy.zeros((l, numdata),dtype=int)
    gamma = numpy.zeros((N, numdata))

    # Create operator
    if operatortype == "randn":
        # generate random operator and normalize
        # TODO: Use sklearn-type random generators instead of numpy.random
        operator = numpy.random.randn(N,n)
        for i in range(operator.shape[0]):
            operator[i,:] = operator[i,:] / numpy.linalg.norm(operator[i,:],2)
    elif operatortype == "orthonormal":
        if n != N:
            raise ValueError("Orthonormal operator has n==N")
        # generate random square operator and orthonormalize
        # TODO: Use sklearn-type random generators instead of numpy.random
        operator = numpy.random.randn(N,n)
        operator = scipy.linalg.orth(operator)
    elif operatortype == "tightframe":
        # random tight frame with normalized rows
        # algorithm from Nam's GAP code
        operator = numpy.random.randn(N,n)
        T = numpy.zeros((N, n))
        tol = 1e-8
        max_j = 200
        j = 1
        while (sum(sum(abs(T-operator))) > numpy.dot(tol,numpy.dot(N,n)) and j < max_j):
            j = j + 1
            T = operator
            [U, S, Vh] = numpy.linalg.svd(operator)
            V = Vh.T
            operatortemp = numpy.dot(numpy.dot(U, numpy.concatenate((numpy.eye(n), numpy.zeros((N-n,n))))), V.transpose())
            operator = numpy.dot(numpy.diag(1.0 / numpy.sqrt(numpy.diag(numpy.dot(operatortemp,operatortemp.transpose())))), operatortemp)
    elif isinstance(operatortype, numpy.ndarray):
        # operator is given
        if N!= operatortype.shape[0] or n != operatortype.shape[1]:
            raise ValueError("Operator shape different from (n,N)")
        operator = operatortype
    else:
        raise ValueError("Wrong operatortype parameter")

    # Generate data from the nullspace of randomly picked l rows
    for i in range(numdata):
        cosupport[:,i] = numpy.sort(numpy.random.permutation(N)[:l])
        [U,D,Vh] = numpy.linalg.svd(operator[cosupport[:,i],:])
        V = Vh.T
        nullspace = V[:,l:]
        data[:,i] = numpy.squeeze(numpy.dot(nullspace, numpy.random.randn(n-l,1)))
        nonzerosupport = numpy.setdiff1d(range(N), cosupport[:,i],True)
        gamma[nonzerosupport,i] = numpy.dot(operator[nonzerosupport,:], data[:,i])

    return data, operator, gamma, cosupport


def make_analysis_compressed_sensing_problem(m, n, N, l, numdata, operator="tightframe", acquisition="randn"):
    """
    Make analysis compressed sensing problem
    """

    # generate cosparse coded data
    data, operator, gamma, cosupport = make_cosparse_coded_signal(n, N, l, numdata, operator)

    # generate acquisition matrix
    if acquisition=="randn":
        acqumatrix = numpy.random.randn(m, n)
    elif isinstance(acquisition, numpy.ndarray):
        # acquisition matrix is given
        if m != acquisition.shape[0] or n != acquisition.shape[1]:
            raise ValueError("Acquisition matrix shape different from (m,n)")
        acqumatrix = acquisition
    else:
        raise ValueError("Unrecognized acquisition matrix type")

    measurements = numpy.dot(acqumatrix, data)

    # TODO: add noise

    return measurements, acqumatrix, data, operator, gamma, cosupport







