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

import generate as gen


class PhaseTransition(with_metaclass(ABCMeta, object)):
    def __init__(self, signaldim, dictdim, deltas, rhos, numdata, solvers):
        self.signaldim = signaldim
        self.dictdim = dictdim
        self.numdata = numdata
        self.deltas = deltas
        self.rhos = rhos
        self.solvers = solvers
        self.avgerr = None
        self.avgERCsuccess = None

    @abstractmethod
    def run(self, solve=True, check=False):
        """
        Runs
        """

    def plot(self, subplot=True, solve=True, check=False, thresh=None):
        #plt.ion() # Turn interactive off

        if solve == False and check==False:
            RuntimeError('Nothing to plot (both solve and check are False)')

        datasources = []

        if solve is True:
            if self.err is None:
                ValueError("No data to plot (have you run()?)")
            else:
                avgerr = dict|()
                for solver in self.err:
                    avgerr[solver] = numpy.zeros(self.err[solver].shape)
                    for idelta in self.err.shape[0]:
                        for irho in self.err.shape[1]:
                            if thresh is None:
                                avgerr[solver][idelta, irho] = np.mean(self.err[solver][idelta, irho])
                            else:
                                avgerr[solver][idelta, irho] = np.count_nonzero(self.err[solver][idelta, irho] < thresh) / self.err[solver][idelta, irho].size
                datasources.append(avgerr)

        if check is True:
            if self.avgERCsuccess is None:
                ValueError("No data to plot (have you check()-ed?)")
            else:
                datasources.append(self.avgERCsuccess)

        for data in datasources:
            if subplot is False:
                pass
            elif subplot is True:
                numsolvers = len(data)
                if numsolvers == 1:
                    subplot = False
                if numsolvers == 2:
                    subplotlayout = (1,2)
                elif numsolvers == 3:
                    subplotlayout = (1,3)
                elif numsolvers == 4:
                    subplotlayout = (2,2)
                elif numsolvers == 5:
                    subplotlayout = (2,3)
                elif numsolvers == 6:
                    subplotlayout = (2,3)
                elif numsolvers == 7:
                    subplotlayout = (2,4)
                elif numsolvers == 8:
                    subplotlayout = (2,4)
                elif numsolvers == 9:
                    subplotlayout = (3,3)
                elif numsolvers == 10:
                    subplotlayout = (2,5)
                else:
                    subplot=False # too many to subplot
            elif len(subplot) == 2:
                subplotlayout = tuple(i for i in subplot)
            else:
                raise ValueError("Incorrect 'subplot' parameter")

            # Don't loop over dictionary, because they are not sorted
            # Better loop on "solvers" list, which preserves order
            #for i, solver in enumerate(data.keys()):
            for i, solver in enumerate(self.solvers):
                if subplot:
                    # simulate column-major order to have
                    #correctpos = i // subplotlayout[0] + (i % subplotlayout[0])*subplotlayout[1]
                    ax = plt.subplot(*(subplotlayout+(i+1,)))
                else:
                    ax = plt.figure()
                plot_phase_transition(data[solver])
                plt.title(solver)
                plt.xlabel(r"$\delta$")
                plt.ylabel(r"$\rho$")

            plt.draw()
        plt.show()

    #TODO: save()

    def computeAverageError(self,shape):
        """
        Returns an array same shape as 'solvers' array, containing the average value of the phase transition
        """
        return np.array([[np.mean(self.avgerr[solver]) for solver in linesolvers]
                         for linesolvers in np.atleast_2d(self.solvers)]).reshape(shape, order='C')

    def plotAverageError(self,shape):
        """
        Plots an array same shape as 'solvers' array, containing the average value of the phase transition
        """
        plt.figure()
        values = self.computeAverageError(shape)
        values = values - np.min(values) # translate minimum to 0
        values = values / np.max(values) # translate maximum to 1
        plot_phase_transition(values, transpose=False)
        plt.show()

class SynthesisPhaseTransition(PhaseTransition):

    """
    Class for running and plotting synthesis-based phase transitions
    """

    def __init__(self, signaldim, dictdim, deltas, rhos, numdata, solvers):
        super(SynthesisPhaseTransition, self).__init__(signaldim, dictdim, deltas, rhos, numdata, solvers)

    def run(self, solve=True, check=False):

        if solve == False and check==False:
            RuntimeError('Nothing to run (both solve and check are False)')

        if solve is True:
            self.avgerr = dict()
            for solver in self.solvers:
                self.avgerr[solver] = np.zeros(shape=(len(self.deltas), len(self.rhos)))
        if check is True:
            self.avgERCsuccess = dict()
            for solver in self.solvers:
                self.avgERCsuccess[solver] = np.zeros(shape=(len(self.deltas), len(self.rhos)))


        for idelta, delta in enumerate(self.deltas):
            for irho, rho in enumerate(self.rhos):
                m = int(round(self.signaldim * delta, 0))  # delta = m/n
                k = int(round(m * rho, 0))                 # rho = k/m

                measurements, acqumatrix, realdata, dictionary, realgamma, realsupport = \
                    gen.make_compressed_sensing_problem(
                        m, self.signaldim, self.dictdim, k, self.numdata, "randn", "randn")

                realdict = {'data':realdata, 'gamma':realgamma, 'support':realsupport}

                for solver in self.solvers:
                    if check is True and hasattr(solver, 'checkERC'):
                        ERCsuccess = solver.checkERC(acqumatrix, dictionary, realsupport)
                        self.avgERCsuccess[solver][idelta, irho] = 1-np.mean(ERCsuccess) # 1- : correct = white
                    if solve is True:
                        gamma = solver.solve(measurements, np.dot(acqumatrix, dictionary), realdict)
                        data = np.dot(dictionary, gamma)
                        errors = data - realdata
                        for i in range(errors.shape[1]):
                            errors[:, i] = errors[:, i] / np.linalg.norm(realdata[:, i])
                            self.err[solver][idelta, irho][i] = np.sqrt(sum(errors**2, 0))
                        #self.avgerr[solver][idelta, irho] = np.mean(np.sqrt(sum(errors**2, 0)))

# TODO: maybe needs refactoring, it's very similar to PhaseTransition()
class AnalysisPhaseTransition(PhaseTransition):
    """
    Class for running and plotting analysis-based phase transitions
    """

    def __init__(self, signaldim, operatordim, deltas, rhos, numdata, solvers):
        super(AnalysisPhaseTransition, self).__init__(signaldim, operatordim, deltas, rhos, numdata, solvers)

    def run(self, solve=True, check=False):

        if solve == False and check==False:
            RuntimeError('Nothing to run (both solve and check are False)')

        if solve is True:
            self.avgerr = dict()
            for solver in self.solvers:
                self.avgerr[solver] = np.zeros(shape=(len(self.deltas), len(self.rhos)))
        if check is True:
            self.avgERCsuccess = dict()
            for solver in self.solvers:
                self.avgERCsuccess[solver] = np.zeros(shape=(len(self.deltas), len(self.rhos)))

        for idelta, delta in enumerate(self.deltas):
            for irho, rho in enumerate(self.rhos):
                m = int(round(self.signaldim * delta, 0))  # delta = m/n
                l = self.signaldim - int(round(m * rho, 0))             # rho = (n-l)/m

                #TODO: make operator only once
                measurements, acqumatrix, realdata, operator, realgamma, realcosupport = \
                    gen.make_analysis_compressed_sensing_problem(
                        m, self.signaldim, self.dictdim, l, self.numdata, "randn", "randn")

                realsupport = np.zeros((operator.shape[0]-realcosupport.shape[0], realcosupport.shape[1]), dtype=int)
                for i in range(realcosupport.shape[1]):
                    realsupport[:,i] = np.setdiff1d(range(operator.shape[0]), realcosupport[:,i])

                for solver in self.solvers:
                    if check is True:
                        ERCsuccess = solver.checkERC(acqumatrix, operator, realsupport)
                        self.avgERCsuccess[solver][idelta, irho] = 1-np.mean(ERCsuccess) # 1- : correct = white
                    if solve is True:
                        data = solver.solve(measurements, acqumatrix, operator)
                        errors = data - realdata
                        for i in range(errors.shape[1]):
                            errors[:, i] = errors[:, i] / np.linalg.norm(realdata[:, i])
                        self.avgerr[solver][idelta, irho] = np.mean(np.sqrt(sum(errors**2, 0)))


# TODO: add many more parameters
def plot_phase_transition(matrix, transpose=True):

    # restrict to [0, 1]
    np.clip(matrix, 0, 1, out=matrix)

    N=1
    # Prepare bigger matrix
    rows, cols = matrix.shape
    bigmatrix = np.zeros((N*rows, N*cols))
    for i in np.arange(rows):
        for j in np.arange(cols):
            bigmatrix[i*N:i*N+N,j*N:j*N+N] = matrix[i,j]

    if transpose:
        bigmatrix = bigmatrix.T

    # plt.figure()
    # Transpose the data so first axis = horizontal, use inverse colormap so small(good) = white, origin = lower left
    plt.imshow(bigmatrix, cmap=cm.gray_r, norm=mcolors.Normalize(0, 1), interpolation='nearest', origin='lower')
