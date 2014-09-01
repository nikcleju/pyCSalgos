"""
phase_transition.py

Class for fast generation of phase-transition graphs

"""

# Author: Nicolae Cleju
# License: BSD 3 clause

from six import with_metaclass
from abc import ABCMeta, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import datetime
import hdf5storage

import generate as gen


class PhaseTransition(with_metaclass(ABCMeta, object)):
    def __init__(self, signaldim, dictdim, deltas, rhos, numdata, solvers):
        self.signaldim = signaldim
        self.dictdim = dictdim
        self.numdata = numdata
        self.deltas = deltas
        self.rhos = rhos
        self.solvers = solvers

        self.err = None
        self.ERCsuccess = None

        self.ERCsolvers = [solver for solver in self.solvers if hasattr(solver, 'checkERC')]
        self.solverNames = [str(solver) for solver in self.solvers]
        self.ERCsolverNames = [str(solver) for solver in self.ERCsolvers]

        self.simData = []

    @abstractmethod
    def run(self, solve=True, check=False):
        """
        Runs
        """

    def clear(self):
        self.simData = []

    def get_description(self):
        """
        Create a long string description of the phase transition
        :return: A string with a logn text description
        """

        # Prepare some strings
        if self.__class__.__name__ == "SynthesisPhaseTransition":
            dictoperstring = "Dictionary"
        elif self.__class__.__name__ == "AnalysisPhaseTransition":
            dictoperstring = "Operator"
        else:
            dictoperstring = "Dictionary/Operator"

        if self.err is None:
            errstr = "None"
        else:
            errstr = "shape " + str(self.err.shape)

        if self.ERCsuccess is None:
            ERCstr = "None"
        else:
            ERCstr = "shape " + str(self.ERCsuccess.shape)

        return ("Date and time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f") + "\n"
        + "Description of phase transition object:\n"
        + str(self) + "\n"
        + "Class type: " + self.__class__.__name__ + "\n"
        + "-------------------\n"
        + "Signal dimension = " + str(self.signaldim) + "\n"
        + dictoperstring + " size = " + str(self.dictdim) + "\n"
        + "Number of signals for each data point = " + str(self.numdata) + "\n"
        + "Delta = " + str(self.deltas) + "\n"
        + "Rho   = " + str(self.rhos) + "\n"
        + "Solvers = " + str(self.solvers) + "\n"
        + "Solvers with Exact Recovery Condition (ERC) = " + str(self.ERCsolvers) + "\n"
        + "Error matrix = " + errstr + "\n"
        + "ERC success matrix = """ + ERCstr + "\n")

    def plot(self, subplot=True, solve=True, check=False, thresh=None, show=True, basename=None, saveexts=None):
        # plt.ion() # Turn interactive off

        if solve is False and check is False:
            RuntimeError('Nothing to plot (both solve and check are False)')

        if show is False and saveexts is None:
            RuntimeError('Neither showing nor saving plot!')

        if basename is None:
            basename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

        datasources = []
        reverse_colormap = []
        datasources_titles = []

        if solve is True:
            if self.err is None:
                ValueError("No data to plot (have you run()?)")
            else:
                datasources.append(self._compute_average(self.err, thresh))
                datasources_titles.append(self.solverNames)
                if thresh is None:
                    reverse_colormap.append(True)
                else:
                    reverse_colormap.append(False)

        if check is True:
            if self.ERCsuccess is None:
                ValueError("No data to plot (have you check()-ed?)")
            else:
                datasources.append(self._compute_average(self.ERCsuccess, thresh=None))
                datasources_titles.append(self.ERCsolverNames)
                reverse_colormap.append(False)  # better means 1, so should not reverse colormap

        for idatasource, data in enumerate(datasources):
            if subplot is False:
                pass
            elif subplot is True:
                numsolvers = len(data)
                if numsolvers == 1:
                    subplot = False
                if numsolvers == 2:
                    subplotlayout = (1, 2)
                elif numsolvers == 3:
                    subplotlayout = (1, 3)
                elif numsolvers == 4:
                    subplotlayout = (2, 2)
                elif numsolvers == 5:
                    subplotlayout = (2, 3)
                elif numsolvers == 6:
                    subplotlayout = (2, 3)
                elif numsolvers == 7:
                    subplotlayout = (2, 4)
                elif numsolvers == 8:
                    subplotlayout = (2, 4)
                elif numsolvers == 9:
                    subplotlayout = (3, 3)
                elif numsolvers == 10:
                    subplotlayout = (2, 5)
                else:
                    subplot = False  # too many to subplot
            elif len(subplot) == 2:
                subplotlayout = tuple(i for i in subplot)
            else:
                raise ValueError("Incorrect 'subplot' parameter")

            # Don't loop over dictionary, because they are not sorted
            # Better loop on "solvers" list, which preserves order
            # for i, solver in enumerate(data.keys()):
            for icurrentdata, currentdata in enumerate(data):
                if subplot:
                    # simulate column-major order to have
                    # correctpos = i // subplotlayout[0] + (i % subplotlayout[0])*subplotlayout[1]
                    plt.subplot(*(subplotlayout + (icurrentdata + 1,)))
                else:
                    plt.figure()
                plot_phase_transition(currentdata, reverse_colormap=reverse_colormap[idatasource])
                plt.title(datasources_titles[idatasource][icurrentdata])
                plt.xlabel(r"$\delta$")
                plt.ylabel(r"$\rho$")

                if not subplot:
                    # separate figure, save each
                    for ext in saveexts:
                        plt.savefig(basename + '_' + str(idatasource) + '_' + str(icurrentdata) + '.' + ext, bbox_inches='tight')
            if subplot:
                # single figure, save at finish
                for ext in saveexts:
                    plt.savefig(basename + '_' + str(idatasource) + '.' + ext, bbox_inches='tight')
            if show:
                plt.draw()
        if show:
            plt.show()

    def _compute_average(self, data, thresh):
        """
        Computes average
        :param data:
        :param thresh:
        :return:
        """
        if thresh is None:
            return np.mean(data, 3)
        else:
            return np.mean(np.abs(data) < thresh, 3)

        # avgerr = []
        # for isolvererr, solvererr in enumerate(data):
        #     current_avgerr = np.zeros(shape=(len(solvererr), len(solvererr[0])))
        #     for idelta in range(current_avgerr.shape[0]):
        #         for irho in range(current_avgerr.shape[1]):
        #             if thresh is None:
        #                 current_avgerr[idelta, irho] = np.mean(solvererr[idelta][irho])
        #             else:
        #                 current_avgerr[idelta, irho] = float(np.count_nonzero(solvererr[idelta][irho] < thresh)) \
        #                                                / len(solvererr[idelta][irho])
        #     avgerr.append(current_avgerr)
        # return avgerr

    def compute_global_average_error(self, shape, thresh=None):
        """
        Returns an array same shape as 'solvers' array, containing the average value of the phase transition
        """
        avgerr = self._compute_average(self.err, thresh=None)
        return np.mean(avgerr, (1,2,3)).reshape(shape, order='C') # mean over all axes except 0, then reshape

    def save(self, basename=None):
        """
        Saves data and parameters to mat file
        :return:
        """
        if basename is None:
            basename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

        # dictionary to save
        mdict = {u'signaldim': self.signaldim, u'dictdim': self.dictdim, u'numdata': self.numdata,
                 u'deltas': self.deltas, u'rhos': self.rhos,
                 u'solverNames': self.solverNames, u'ERCsolverNames': self.ERCsolverNames,
                 u'err': self.err, u'ERCsuccess': self.ERCsuccess, u'simData': self.simData,
                 u'description': self.get_description()}

        hdf5storage.savemat(basename + '.mat', mdict)
        with open(basename+".txt", "w") as f:
            f.write(self.get_description())

    def load(self, filename):
        """
        Loads data from saved file.
        A subsequent run() will overwrite the data. Currently no option to rerun same data (but planned).
        :param filename:
        :return:
        """

        mdict = hdf5storage.loadmat(filename)
        self.signaldim = mdict[u'signaldim']
        self.dictdim = mdict[u'dictdim']
        self.numdata = mdict[u'numdata']
        self.deltas = mdict[u'deltas']
        self.rhos = mdict[u'rhos']
        self.err = mdict[u'err']
        self.ERCsuccess = mdict[u'ERCsuccess']
        self.simData = mdict[u'simData']



    def plot_average_error(self, shape, thresh=None):
        """
        Plots an array same shape as 'solvers' array, containing the average value of the phase transition
        """
        plt.figure()
        values = self.compute_global_average_error(shape, thresh)
        values = values - np.min(values)  # translate minimum to 0
        values = values / np.max(values)  # translate maximum to 1
        plot_phase_transition(values, transpose=False)
        plt.show()


class SynthesisPhaseTransition(PhaseTransition):
    """
    Class for running and plotting synthesis-based phase transitions
    """

    def __init__(self, signaldim, dictdim, deltas, rhos, numdata, solvers):
        super(SynthesisPhaseTransition, self).__init__(signaldim, dictdim, deltas, rhos, numdata, solvers)

    def run(self, solve=True, check=False):

        if solve is False and check is False:
            RuntimeError('Nothing to run (both solve and check are False)')

        # Initialize zero-filled arrays
        if solve is True:
            #self.err = [np.zeros(shape=(len(self.deltas), len(self.rhos), self.numdata)) for _ in self.solvers]
            self.err = np.zeros(shape=(len(self.solvers), len(self.deltas), len(self.rhos), self.numdata))
        if check is True:
            #self.ERCsuccess = [np.zeros(shape=(len(self.deltas), len(self.rhos), self.numdata), dtype=bool)
            #                   for _ in self.ERCsolvers]
            self.ERCsuccess = \
                np.zeros(shape=(len(self.ERCsolvers), len(self.deltas), len(self.rhos), self.numdata), dtype=bool)

        if not self.simData:
            # a 2D list of dictionaries, size deltas x rhos
            self.simData = [[dict() for _ in self.rhos] for _ in self.deltas]

        for idelta, delta in enumerate(self.deltas):
            for irho, rho in enumerate(self.rhos):
                m = int(round(self.signaldim * delta, 0))  # delta = m/n
                k = int(round(m * rho, 0))  # rho = k/m

                if not self.simData[idelta][irho]:
                    measurements, acqumatrix, realdata, dictionary, realgamma, realsupport = \
                        gen.make_compressed_sensing_problem(
                            m, self.signaldim, self.dictdim, k, self.numdata, "randn", "randn")
                    self.simData[idelta][irho][u'measurements'] = measurements
                    self.simData[idelta][irho][u'acqumatrix'] = acqumatrix
                    self.simData[idelta][irho][u'realdata'] = realdata
                    self.simData[idelta][irho][u'dictionary'] = dictionary
                    self.simData[idelta][irho][u'realgamma'] = realgamma
                    self.simData[idelta][irho][u'realsupport'] = realsupport
                else:
                    measurements = self.simData[idelta][irho][u'measurements']
                    acqumatrix = self.simData[idelta][irho][u'acqumatrix']
                    realdata = self.simData[idelta][irho][u'realdata']
                    dictionary = self.simData[idelta][irho][u'dictionary']
                    realgamma = self.simData[idelta][irho][u'realgamma']
                    realsupport = self.simData[idelta][irho][u'realsupport']

                realdict = {'data': realdata, 'gamma': realgamma, 'support': realsupport}

                if check is True:
                    for iERCsolver, ERCsolver in enumerate(self.ERCsolvers):
                        self.ERCsuccess[iERCsolver, idelta, irho] = ERCsolver.checkERC(acqumatrix, dictionary,
                                                                                       realsupport)
                if solve is True:
                    for isolver, solver in enumerate(self.solvers):
                        gamma = solver.solve(measurements, np.dot(acqumatrix, dictionary), realdict)
                        data = np.dot(dictionary, gamma)
                        errors = data - realdata
                        for i in range(errors.shape[1]):
                            errors[:, i] = errors[:, i] / np.linalg.norm(realdata[:, i])
                            self.err[isolver, idelta, irho][i] = np.sqrt(sum(errors[:, i] ** 2))


# TODO: maybe needs refactoring, it's very similar to PhaseTransition()
class AnalysisPhaseTransition(PhaseTransition):
    """
    Class for running and plotting analysis-based phase transitions
    """

    def __init__(self, signaldim, operatordim, deltas, rhos, numdata, solvers):
        super(AnalysisPhaseTransition, self).__init__(signaldim, operatordim, deltas, rhos, numdata, solvers)

    def run(self, solve=True, check=False):

        if solve is False and check is False:
            RuntimeError('Nothing to run (both solve and check are False)')

        # Initialize zero-filled arrays
        if solve is True:
            #self.err = [np.zeros(shape=(len(self.deltas), len(self.rhos), self.numdata)) for _ in self.solvers]
            self.err = np.zeros(shape=(len(self.solvers), len(self.deltas), len(self.rhos), self.numdata))
        if check is True:
            #self.ERCsuccess = [np.zeros(shape=(len(self.deltas), len(self.rhos), self.numdata), dtype=bool)
            #                   for _ in self.ERCsolvers]
            self.ERCsuccess = \
                np.zeros(shape=(len(self.ERCsolvers), len(self.deltas), len(self.rhos), self.numdata), dtype=bool)

        if not self.simData:
            # a 2D list of dictionaries, size deltas x rhos
            self.simData = [[dict() for _ in self.rhos] for _ in self.deltas]

        for idelta, delta in enumerate(self.deltas):
            for irho, rho in enumerate(self.rhos):
                m = int(round(self.signaldim * delta, 0))  # delta = m/n
                l = self.signaldim - int(round(m * rho, 0))  # rho = (n-l)/m
                # l = int(round(self.signaldim - m * rho, 0))  #TODO: DEBUG, TO CHECK IF THE SAME!!

                if not self.simData[idelta][irho]:
                    measurements, acqumatrix, realdata, operator, realgamma, realcosupport = \
                        gen.make_analysis_compressed_sensing_problem(
                            m, self.signaldim, self.dictdim, l, self.numdata, operator="randn", acquisition="randn")
                    self.simData[idelta][irho][u'measurements'] = measurements
                    self.simData[idelta][irho][u'acqumatrix'] = acqumatrix
                    self.simData[idelta][irho][u'realdata'] = realdata
                    self.simData[idelta][irho][u'operator'] = operator
                    self.simData[idelta][irho][u'realgamma'] = realgamma
                    self.simData[idelta][irho][u'realcosupport'] = realcosupport
                    #self.simData[idelta][irho][u'realsupport'] = realsupport
                else:
                    measurements = self.simData[idelta][irho][u'measurements']
                    acqumatrix = self.simData[idelta][irho][u'acqumatrix']
                    realdata = self.simData[idelta][irho][u'realdata']
                    operator = self.simData[idelta][irho][u'operator']
                    realgamma = self.simData[idelta][irho][u'realgamma']
                    realcosupport = self.simData[idelta][irho][u'realcosupport']

                # TODO: DEBUG HACK:
                # savename = 'signals_delta' + str(delta) + '_rho' + str(rho) + ".mat"
                # mdict = scipy.io.loadmat(savename)#, {'operator':Omega, 'measurements':y,'acqumatrix':M, 'realdata':x0})
                # measurements = mdict['measurements']
                # acqumatrix = mdict['acqumatrix']
                # realdata = mdict['realdata']
                # operator = mdict['operator']
                # realcosupport = mdict['realcosupport']

                realdict = {'data': realdata, 'gamma': realgamma, 'cosupport': realcosupport}

                realsupport = np.zeros((operator.shape[0] - realcosupport.shape[0], realcosupport.shape[1]), dtype=int)
                for i in range(realcosupport.shape[1]):
                    realsupport[:, i] = np.setdiff1d(range(operator.shape[0]), realcosupport[:, i])

                if check is True:
                    for iERCsolver, ERCsolver in enumerate(self.ERCsolvers):
                        self.ERCsuccess[iERCsolver, idelta, irho] = ERCsolver.checkERC(acqumatrix, operator,
                                                                                       realsupport)
                if solve is True:
                    for isolver, solver in enumerate(self.solvers):
                        data = solver.solve(measurements, acqumatrix, operator, realdict)
                        errors = data - realdata
                        for i in range(errors.shape[1]):
                            errors[:, i] = errors[:, i] / np.linalg.norm(realdata[:, i])
                            self.err[isolver, idelta, irho][i] = np.sqrt(sum(errors[:, i] ** 2))


# TODO: add many more parameters
def plot_phase_transition(matrix, transpose=True, reverse_colormap=False):
    # restrict to [0, 1]
    np.clip(matrix, 0, 1, out=matrix)

    N = 1
    # Prepare bigger matrix
    rows, cols = matrix.shape
    bigmatrix = np.zeros((N * rows, N * cols))
    for i in np.arange(rows):
        for j in np.arange(cols):
            bigmatrix[i * N:i * N + N, j * N:j * N + N] = matrix[i, j]

    if transpose:
        bigmatrix = bigmatrix.T

    # plt.figure()
    # Transpose the data so first axis = horizontal, use inverse colormap so small(good) = white, origin = lower left
    if reverse_colormap:
        cmap = cm.gray_r
    else:
        cmap = cm.gray
    plt.imshow(bigmatrix, cmap=cmap, norm=mcolors.Normalize(0, 1), interpolation='nearest', origin='lower')
