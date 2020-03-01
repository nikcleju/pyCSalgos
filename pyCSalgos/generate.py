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
except ImportError as e:
    # module doesn't exist
    has_sklearn_datasets = False

def add_noise_snr(data, snr_db, rng=None):
    """
    Adda a certain amount of noise over some data.
    The amount of noise is specified in SNR [db]
    """
    if rng is not None:
        # Add noise on data
        if numpy.isfinite(snr_db):
            noise = rng.randn(*data.shape)
            SNR_norm = 10**(snr_db/20.)
            for i in range(data.shape[1]):
                # Make norm 1
                noise[:,i] = noise[:,i] / numpy.linalg.norm(noise[:,i])
                # Make noise norm = smaller than data by SNR_norm
                noise[:,i] = noise[:,i] * numpy.linalg.norm(data[:,i]) / SNR_norm
        else:
            noise = 0
    
    return data + noise

def make_sparse_coded_signal(signal_size, dict_size, sparsity, num_data, snr_db_sparse, snr_db_signal, dictionary="randn",
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
    snr_db : float
        Signal to Noise Ratio (dB). Can be numpy.inf for no noise.
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

    if isinstance(dictionary, str) and dictionary == "randn" and use_sklearn and has_sklearn_datasets:
        # use random normalized dictionary from scikit-learn
        data, dictionary, gamma = sklearn.datasets.make_sparse_coded_signal(n_samples=num_data, n_features=signal_size,
                                                                n_components=dict_size ,n_nonzero_coefs=sparsity,
                                                                random_state=rng)
        for i in range(num_data):
            support[:, i] = numpy.nonzero(gamma[:, i])[0]

    else:
        # Create dictionary
        if isinstance(dictionary, str) and dictionary == "randn":
            # generate random dictionary and normalize
            dictionary = rng.randn(signal_size, dict_size)
            dictionary = dictionary / numpy.sqrt(numpy.sum(dictionary**2, axis=0))
        elif isinstance(dictionary, str) and dictionary == "orthonormal":
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

    # Add sparsity noise (noise on the decomposition vector)
    cleargamma = gamma.copy()  # sparse gamma with no noise
    gamma = add_noise_snr(gamma, snr_db_sparse, rng)
    data = numpy.dot(dictionary, gamma)

    # Add signal noise
    # if numpy.isfinite(snr_db_signal):
    #     noise = rng.randn(signal_size, num_data)
    #     SNR_norm = 10**(snr_db_signal/20.)
    #     for i in range(num_data):
    #         # Make norm 1
    #         noise[:,i] = noise[:,i] / numpy.linalg.norm(noise[:,i])
    #         # Make smaller than data by SNR_norm
    #         noise[:,i] = noise[:,i] / SNR_norm * numpy.linalg.norm(data[:,i])
    #         #(numpy.linalg.norm(data[:,i])**2 / numpy.linalg.norm(noise[:,i]))
    # else:
    #     noise = 0
    cleardata = data.copy() # data with no signal noise
    data = add_noise_snr(data, snr_db_signal, rng)
    #data = data + noise

    return data,dictionary,gamma,support,cleardata


def make_compressed_sensing_problem(num_measurements, signal_size, dict_size, sparsity, num_data, snr_db_sparse, snr_db_signal, snr_db_meas,
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
    snr_db_sparse : float
        Signal to Noise Ratio (dB) of the sparse decomposition. Can be numpy.inf for no noise.
    snr_db_signal : float
        Signal to Noise Ratio (dB) of the signal (dict * decomposition). Can be numpy.inf for no noise.
    snr_db_meas: float
        Signal to Noise Ratio (dB) of the measurements. Can be numpy.inf for no noise.
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
    data, dictionary, gamma, support, cleardata = make_sparse_coded_signal(signal_size, dict_size, sparsity ,num_data, snr_db_sparse, snr_db_sparse,
                                                                dictionary, use_sklearn, random_state=rng)

    # generate acquisition matrix
    if isinstance(acquisition, str) and acquisition == "randn":
        acqumatrix = rng.randn(num_measurements, signal_size)
    elif isinstance(acquisition, numpy.ndarray):
        # acquisition matrix is given
        if num_measurements != acquisition.shape[0] or signal_size != acquisition.shape[1]:
            raise ValueError("Acquisition matrix shape different from (m,n)")
        acqumatrix = acquisition
    else:
        raise ValueError("Unrecognized acquisition matrix type")

    measurements = numpy.dot(acqumatrix, data)

    # Add measurement noise
    measurements = add_noise_snr(measurements, snr_db_meas, rng)

    return measurements, acqumatrix, data, dictionary, gamma, support, cleardata



def make_cosparse_coded_signal(signal_size, operator_size, cosparsity, num_data, snr_db, operator="tightframe",
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
    snr_db : float
        Signal to Noise Ratio (dB). Can be numpy.inf for no noise.
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
    bNonZerosupport = numpy.zeros((operator_size, num_data), dtype=bool)
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
            [U, S, Vh] = numpy.linalg.svd(operator, full_matrices=False)
            V = Vh.T
            # U * In*0n * VT
            #operatortemp = numpy.dot(numpy.dot(U, numpy.concatenate((numpy.eye(signal_size), numpy.zeros((operator_size-signal_size,signal_size))))), V.transpose())
            operatortemp = numpy.dot(U, V.transpose())
            #assert(numpy.linalg.norm(operatortemp - operatortemp2) < 1e-16)
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

        # ! SVD is very slow, use QR decomposition instead
        #[U,D,Vh] = numpy.linalg.svd(operator[cosupport[:,i],:])
        #V = Vh.T
        #nullspace = V[:,cosparsity:]

        Q, _ = scipy.linalg.qr(operator[cosupport[:,i],:].T)
        nullspace = Q[:, cosparsity:]

        data[:,i] = numpy.squeeze(numpy.dot(nullspace, rng.randn(signal_size-cosparsity,1)))

        nonzerosupport = numpy.setdiff1d(range(operator_size), cosupport[:,i],True)
        bNonZerosupport[nonzerosupport,i] = True

    #gamma[nonzerosupport, i] = numpy.dot(operator[nonzerosupport,:], data[:,i])
    gamma[bNonZerosupport] = numpy.dot(operator, data)[bNonZerosupport]

    # Add noise
    if numpy.isfinite(snr_db):
        noise = rng.randn(signal_size, num_data)
        SNR_norm = 10**(snr_db/20.)
        for i in range(num_data):
            # Make norm 1
            noise[:,i] = noise[:,i] / numpy.linalg.norm(noise[:,i])
            # Make smaller than data by SNR_norm
            noise[:,i] = noise[:,i] / SNR_norm * numpy.linalg.norm(data[:,i])
            (numpy.linalg.norm(data[:,i])**2 / numpy.linalg.norm(noise[:,i]))
    else:
        noise = 0
    cleardata = data.copy() # no noise data
    data = data + noise

    return data, operator, gamma, cosupport, cleardata


def make_analysis_compressed_sensing_problem(num_measurements, signal_size, operator_size, cosparsity, num_data, snr_db,
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
    snr_db : float
        Signal to Noise Ratio (dB). Can be numpy.inf for no noise.
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
    data, operator, gamma, cosupport, cleardata = make_cosparse_coded_signal(signal_size, operator_size, cosparsity,
                                                                  num_data, snr_db, operator, random_state=rng)

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

    return measurements, acqumatrix, data, operator, gamma, cosupport, cleardata







