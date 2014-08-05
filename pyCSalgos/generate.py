"""
Contains function to generate datasets and problems
"""

# Author: Nicolae Cleju
# License: BSD 3 clause

import numpy
import scipy
from sklearn.utils import check_random_state

try:
    import sklearn.datasets
    has_sklearn_datasets = True
except ImportError, e:
    # module doesn't exist
    has_sklearn_datasets = False


def make_sparse_coded_signal(signal_size, dict_size, sparsity, num_data, dictionary="randn",
                             use_sklearn=True, random_state=None):
    """
    Generate sparse coded signals.

    Parameters
    ----------
    signal_size : int
        Signal dimension.
    dict_size : int
        Dictionary dimension.
    sparsity : int
        Desired sparsity of the signal.
    num_data : int
        Number of signals to generate.
    dictionary : {'randn', 'orthonormal', a numpy matrix}, optional (default="randn")
         The type of dictionary. Can be one of the following:
        - "randn" (default): i.i.d. random gaussian entries, atoms (columns) are normalized
        - "orthonormal": a random orthonormal matrix
        - a numpy matrix that will be used.
    use_sklearn : boolean, optional (default=True)
        If true (default), use the corresponding function from the scikit-learn package, if available.
        If false or scikit-learn not available, use the local similar version.
    random_state : int or RandomState instance, optional (default=None)
        Set random number generator state.

    Returns
    -------
    data : array_like
        The sparse signal(s), as a vector or a (signal_size x num_data) matrix containing the sparse signals as columns.
    dictionary : array_like
        The dictionary matrix, size (signal_size x dict_size)
    gamma :
        The sparse codes themselves, size (dict_size x num_data)
    support :
        The locations of the non-zeros in ``gamma'', size (sparsity x num_data)
    """

    rng = check_random_state(random_state)

    # Generate coefficients matrix
    support = numpy.zeros((sparsity, num_data), dtype=int)

    if dictionary == "randn" and use_sklearn and has_sklearn_datasets:
        # use random normalized dictionary from scikit-learn
        data, dictionary, gamma = sklearn.datasets.make_sparse_coded_signal(n_samples=num_data, n_features=signal_size,
                                                                n_components=dict_size ,n_nonzero_coefs=sparsity,
                                                                random_state=rng)
        for i in range(num_data):
            support[:, i] = numpy.nonzero(gamma[:, i])[0]

    else:
        # Create dictionary
        if dictionary == "randn":
            # generate random dictionary and normalize
            dictionary = rng.randn(signal_size, dict_size)
            dictionary = dictionary / numpy.sqrt(numpy.sum(dictionary**2, axis=0))
        elif dictionary == "orthonormal":
            if signal_size != dict_size:
                raise ValueError("Orthonormal dictionary has n==N")
            # generate random square dictionary and orthonormalize
            dictionary = rng.randn(signal_size,dict_size)
            dictionary = scipy.linalg.orth(dictionary)
        elif isinstance(dictionary, numpy.ndarray):
            # dictionary is given
            if signal_size != dictionary.shape[0] or dict_size != dictionary.shape[1]:
                raise ValueError("Dictionary shape different from (n,N)")
            dictionary = dictionary
        else:
            raise ValueError("Wrong dictionary parameter")

        # Generate coefficients matrix
        gamma = numpy.zeros((dict_size, num_data))
        for i in range(num_data):
            support[:, i] = rng.permutation(dict_size)[:sparsity]
            gamma[support[:, i],i] = rng.randn(sparsity)

        # Generate data
        data = numpy.dot(dictionary,gamma)

    return data,dictionary,gamma,support


def make_compressed_sensing_problem(num_measurements, signal_size, dict_size, sparsity, num_data,
                                    dictionary="randn", acquisition="randn", use_sklearn=True, random_state=None):
    """
    Generate a random compressed sensing problem.

    Parameters
    ----------
    num_measurements : int
        Number of measurements.
    signal_size : int
        Signal dimension.
    dict_size : int
        Dictionary dimension.
    sparsity : int
        Desired sparsity of the signal.
    num_data : int
        Number of signals to generate.
    dictionary : {'randn', 'orthonormal', a numpy matrix}, optional (default="randn")
         The type of dictionary. Can be one of the following:
        - "randn" (default): i.i.d. random gaussian entries, atoms (columns) are normalized
        - "orthonormal": a random orthonormal matrix
        - a numpy matrix that will be used.
    acquisition : {'randn', a numpy matrix}, optional (default="randn")
         The type of acquisition. Can be one of the following:
        - "randn" (default): i.i.d. random gaussian entries
        - a numpy matrix that will be used as acquisition matrix.
    use_sklearn : boolean, optional (default=True)
        Argument passed to ``make_sparse_coded_signal()''
        If true (default), use the function ``make_sparse_coded_signal()'' from the scikit-learn package
        to create the sparse coded data.
        If false or scikit-learn not available, use the local similar version.
    random_state : int or RandomState instance, optional (default=None)
        Set random number generator state.

    Returns
    -------
    measurements : array_like
        The measurement vector/matrix, size (num_measurements x num_data)
    acqumatrix : array_like
        The acquisition matrix, size (num_measurements x signal_size)
    data : array_like
        The sparse signal(s), as a vector or a (signal_size x num_data) matrix
         containing the sparse signals as columns.
    dictionary : array_like
        The dictionary matrix, size (signal_size x dict_size)
    gamma :
        The sparse codes themselves, size (dict_size x num_data)
    support :
        The locations of the non-zeros in ``gamma'', size (sparsity x num_data)
    """

    rng = check_random_state(random_state)

    # generate sparse coded data
    data, dictionary, gamma, support = make_sparse_coded_signal(signal_size, dict_size, sparsity ,num_data,
                                                                dictionary, use_sklearn, random_state=rng)

    # generate acquisition matrix
    if acquisition == "randn":
        acqumatrix = rng.randn(num_measurements, signal_size)
    elif isinstance(acquisition, numpy.ndarray):
        # acquisition matrix is given
        if num_measurements != acquisition.shape[0] or signal_size != acquisition.shape[1]:
            raise ValueError("Acquisition matrix shape different from (m,n)")
        acqumatrix = acquisition
    else:
        raise ValueError("Unrecognized acquisition matrix type")

    measurements = numpy.dot(acqumatrix, data)

    return measurements, acqumatrix, data, dictionary, gamma, support



def make_cosparse_coded_signal(signal_size, operator_size, cosparsity, num_data, operator="tightframe",
                               random_state=None):
    """
    Generate co-sparse coded signals

    Parameters
    ----------
    signal_size : int
        Signal dimension.
    operator_size : int
        Operator dimension.
    cosparsity : int
        Desired cosparsity of the signal.
    num_data : int
        Number of signals to generate.
    operator : {'tightframe', 'randn', 'orthonormal', a numpy matrix}, optional (default="tightframe")
         The type of operator. Can be one of the following:
        - "tightframe" (default): a random tight frame (tall matrix), with normalized rows
        - "randn": i.i.d. random gaussian entries, atoms (rows) are normalized
        - "orthonormal": a random orthonormal matrix
        - a numpy matrix that will be used as operator matrix
    random_state : int or RandomState instance, optional (default=None)
        Set random number generator state.

    Returns
    -------
    data : array_like
        The cosparse signal(s), as a vector or a (signal_size x num_data) matrix containing the signals as columns.
    operator : array_like
        The operator matrix, size (operator_size x signal_size)
    gamma :
        The sparse codes themselves, size (operator_size x num_data)
    cosupport :
        The locations of the zeros in ``gamma'', size (cosparsity x num_data)
    """

    rng = check_random_state(random_state)

    # Prepare matrices
    data = numpy.zeros((signal_size, num_data))
    cosupport = numpy.zeros((cosparsity, num_data), dtype=int)
    gamma = numpy.zeros((operator_size, num_data))

    # Create operator
    if operator == "randn":
        # generate random operator and normalize
        operator = rng.randn(operator_size,signal_size)
        for i in range(operator.shape[0]):
            operator[i,:] = operator[i,:] / numpy.linalg.norm(operator[i,:],2)
    elif operator == "orthonormal":
        if signal_size != operator_size:
            raise ValueError("Orthonormal operator has n==N")
        # generate random square operator and orthonormalize
        operator = rng.randn(operator_size,signal_size)
        operator = scipy.linalg.orth(operator)
    elif operator == "tightframe":
        # random tight frame with normalized rows
        # algorithm from Nam's GAP code
        operator = rng.randn(operator_size,signal_size)
        T = numpy.zeros((operator_size, signal_size))
        tol = 1e-8
        max_j = 200
        j = 1
        while (sum(sum(abs(T-operator))) > numpy.dot(tol,numpy.dot(operator_size,signal_size)) and j < max_j):
            j = j + 1
            T = operator
            [U, S, Vh] = numpy.linalg.svd(operator)
            V = Vh.T
            operatortemp = numpy.dot(numpy.dot(U, numpy.concatenate((numpy.eye(signal_size), numpy.zeros((operator_size-signal_size,signal_size))))), V.transpose())
            operator = numpy.dot(numpy.diag(1.0 / numpy.sqrt(numpy.diag(numpy.dot(operatortemp,operatortemp.transpose())))), operatortemp)
    elif isinstance(operator, numpy.ndarray):
        # operator is given
        if operator_size!= operator.shape[0] or signal_size != operator.shape[1]:
            raise ValueError("Operator shape different from (n,N)")
        operator = operator
    else:
        raise ValueError("Wrong operatortype parameter")

    # Generate data from the nullspace of randomly picked l rows
    for i in range(num_data):
        cosupport[:,i] = numpy.sort(rng.permutation(operator_size)[:cosparsity])
        [U,D,Vh] = numpy.linalg.svd(operator[cosupport[:,i],:])
        V = Vh.T
        nullspace = V[:,cosparsity:]
        data[:,i] = numpy.squeeze(numpy.dot(nullspace, rng.randn(signal_size-cosparsity,1)))
        nonzerosupport = numpy.setdiff1d(range(operator_size), cosupport[:,i],True)
        gamma[nonzerosupport,i] = numpy.dot(operator[nonzerosupport,:], data[:,i])

    return data, operator, gamma, cosupport


def make_analysis_compressed_sensing_problem(num_measurements, signal_size, operator_size, cosparsity, num_data,
                                             operator="tightframe", acquisition="randn", random_state=None):
    """
    Generate a random analysis compressed sensing problem

    Parameters
    ----------
    num_measurements : int
        Number of measurements.
    signal_size : int
        Signal dimension.
    operator_size : int
        Operator dimension.
    cosparsity : int
        Desired cosparsity of the signal.
    num_data : int
        Number of signals to generate.
    operator : {'tightframe', 'randn', 'orthonormal', a numpy matrix}, optional (default="tightframe")
         The type of operator. Can be one of the following:
        - "tightframe" (default): a random tight frame (tall matrix), with normalized rows
        - "randn": i.i.d. random gaussian entries, atoms (rows) are normalized
        - "orthonormal": a random orthonormal matrix
        - a numpy matrix that will be used as operator matrix
    acquisition : {'randn', a numpy matrix}, optional (default="randn")
         The type of acquisition. Can be one of the following:
        - "randn" (default): i.i.d. random gaussian entries
        - a numpy matrix that will be used as acquisition matrix.
    random_state : int or RandomState instance, optional (default=None)
        Set random number generator state.

    Returns
    -------
    measurements : array_like
        The measurement vector/matrix, size (num_measurements x num_data)
    acqumatrix : array_like
        The acquisition matrix, size (num_measurements x signal_size)
    data : array_like
        The cosparse signal(s), as a vector or a (signal_size x num_data) matrix
         containing the signals as columns.
    operator : array_like
        The operator matrix, size (operator_size x signal_size)
    gamma :
        The sparse codes themselves, size (operator_size x num_data)
    cosupport :
        The locations of the zeros in ``gamma'', size (sparsity x num_data)
    """

    rng = check_random_state(random_state)

    # generate cosparse coded data
    data, operator, gamma, cosupport = make_cosparse_coded_signal(signal_size, operator_size, cosparsity, num_data, operator, random_state=rng)

    # generate acquisition matrix
    if acquisition=="randn":
        acqumatrix = rng.randn(num_measurements, signal_size)
    elif isinstance(acquisition, numpy.ndarray):
        # acquisition matrix is given
        if num_measurements != acquisition.shape[0] or signal_size != acquisition.shape[1]:
            raise ValueError("Acquisition matrix shape different from (m,n)")
        acqumatrix = acquisition
    else:
        raise ValueError("Unrecognized acquisition matrix type")

    measurements = numpy.dot(acqumatrix, data)

    # TODO: add noise

    return measurements, acqumatrix, data, operator, gamma, cosupport







