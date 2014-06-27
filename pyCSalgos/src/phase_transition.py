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

    @abstractmethod
    def run(self):
        """
        Runs
        """

    def plot(self, subplot=True):
        if self.avgerr is None:
            ValueError("No data to plot (have you run()?)")

        if subplot is False:
            pass
        elif subplot is True:
            numsolvers = len(self.avgerr)
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
        elif len(subplot) == 2:
            subplotlayout = tuple(i for i in subplot)
        else:
            raise ValueError("Incorrect 'subplot' parameter")

        for i, solver in enumerate(self.avgerr.keys()):
            if subplot is True:
                plt.subplot(*(subplotlayout+(i,)))
            else:
                plt.figure()
            plot_phase_transition(self.avgerr[solver])
            plt.title(solver)
            plt.xlabel(r"$\delta$")
            plt.ylabel(r"$\rho$")

        plt.show()

    #TODO: save()


class SynthesisPhaseTransition(PhaseTransition):
    """
    Class for running and plotting synthesis-based phase transitions
    """

    def __init__(self, signaldim, dictdim, deltas, rhos, numdata, solvers):
        super(SynthesisPhaseTransition, self).__init__(signaldim, dictdim, deltas, rhos, numdata, solvers)

    def run(self):
        self.avgerr = dict()
        for solver in self.solvers:
            self.avgerr[solver] = np.zeros(shape=(len(self.deltas), len(self.rhos)))

        for idelta, delta in enumerate(self.deltas):
            for irho, rho in enumerate(self.rhos):
                m = int(round(self.signaldim * delta, 0))  # delta = m/n
                k = int(round(m * rho, 0))                 # rho = k/m

                measurements, acqumatrix, realdata, dictionary, realgamma, realsupport = \
                    gen.make_compressed_sensing_problem(
                        m, self.signaldim, self.dictdim, k, self.numdata, "randn", "randn")

                for solver in self.solvers:
                    gamma = solver.solve(measurements, np.dot(acqumatrix, dictionary))
                    data = np.dot(dictionary, gamma)

                    errors = data - realdata
                    for i in range(errors.shape[1]):
                        errors[:, i] = errors[:, i] / np.linalg.norm(realdata[:, i])
                    self.avgerr[solver][idelta, irho] = np.mean(np.sqrt(sum(errors**2, 0)))


# TODO: maybe needs refactoring, it's very similar to PhaseTransition()
class AnalysisPhaseTransition(PhaseTransition):
    """
    Class for running and plotting analysis-based phase transitions
    """

    def __init__(self, signaldim, operatordim, deltas, rhos, numdata, solvers):
        super(AnalysisPhaseTransition, self).__init__(signaldim, operatordim, deltas, rhos, numdata, solvers)

    def run(self):
        self.avgerr = dict()
        for solver in self.solvers:
            self.avgerr[solver] = np.zeros(shape=(len(self.deltas), len(self.rhos)))

        for idelta, delta in enumerate(self.deltas):
            for irho, rho in enumerate(self.rhos):
                m = int(round(self.signaldim * delta, 0))  # delta = m/n
                l = self.signaldim - int(round(m * rho, 0))             # rho = (n-l)/m

                #TODO: make operator only once
                measurements, acqumatrix, realdata, operator, realgamma, realcosupport = \
                    gen.make_analysis_compressed_sensing_problem(
                        m, self.signaldim, self.dictdim, l, self.numdata, "randn", "randn")

                for solver in self.solvers:
                    data = solver.solve(measurements, acqumatrix, operator)

                    errors = data - realdata
                    for i in range(errors.shape[1]):
                        errors[:, i] = errors[:, i] / np.linalg.norm(realdata[:, i])
                    self.avgerr[solver][idelta, irho] = np.mean(np.sqrt(sum(errors**2, 0)))


# TODO: add many more parameters
def plot_phase_transition(matrix):

    # restrict to [0, 1]
    np.clip(matrix, 0, 1, out=matrix)

    N=1
    # Prepare bigger matrix
    rows, cols = matrix.shape
    bigmatrix = np.zeros((N*rows, N*cols))
    for i in np.arange(rows):
        for j in np.arange(cols):
            bigmatrix[i*N:i*N+N,j*N:j*N+N] = matrix[i,j]

    # plt.figure()
    # Transpose the data so first axis = horizontal, use inverse colormap so small(good) = white, origin = lower left
    plt.imshow(bigmatrix.T, cmap=cm.gray_r, norm=mcolors.Normalize(0, 1), interpolation='nearest', origin='lower')

