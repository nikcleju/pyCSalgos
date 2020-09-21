"""
phase_transition.py

Class for fast generation of phase-transition graphs

"""

# Author: Nicolae Cleju
# License: BSD 3 clause

from six import with_metaclass
from abc import ABCMeta, abstractmethod
import types

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.ticker import FormatStrFormatter

import datetime
import hdf5storage
import pickle as cPickle   # Python3 has no cPickle
import multiprocessing

from . import generate as gen


class PhaseTransition(with_metaclass(ABCMeta, object)):
    def __init__(self, signaldim, dictdim, deltas, rhos, numdata, snr_db_sparse, snr_db_signal, snr_db_meas, solvers=[]):

        self.signaldim = signaldim
        self.dictdim = dictdim
        self.numdata = numdata
        self.deltas = deltas
        self.rhos = rhos
        self.snr_db_sparse = snr_db_sparse
        self.snr_db_signal = snr_db_signal
        self.snr_db_meas   = snr_db_meas

        self.err = None
        self.ERCsuccess = None
        self.gamma = None
        self.support = None
        self.simData = []

        self.solvers = solvers
        self.ERCsolvers = [solver for solver in self.solvers if hasattr(solver, 'checkERC')]
        self.solverNames = [str(solver) for solver in self.solvers]
        self.ERCsolverNames = [str(solver) for solver in self.ERCsolvers]

        self.clear()

    @abstractmethod
    def run(self, solve=True, check=False):
        """
        Runs
        """

    def clear(self):
        self.err = None
        self.ERCsuccess = None
        self.gamma = None
        self.support = None        
        self.simData = []

    def set_solvers(self, solvers):
        self.clear()
        self.solvers = solvers
        self.ERCsolvers = [solver for solver in self.solvers if hasattr(solver, 'checkERC')]
        self.solverNames = [str(solver) for solver in self.solvers]
        self.ERCsolverNames = [str(solver) for solver in self.ERCsolvers]

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

        if self.gamma is None:
            gammastr = "None"
        else:
            gammastr = "shape " + str(self.gamma.shape)

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
        + "SNR_Sparse = " + str(self.snr_db_sparse) + " dB\n"
        + "SNR_Signal = " + str(self.snr_db_signal) + " dB\n"
        + "SNR_Meas = " + str(self.snr_db_meas) + " dB\n"
        + "Solvers = " + str(self.solvers) + "\n"
        + "Solvers with Exact Recovery Condition (ERC) = " + str(self.ERCsolvers) + "\n"
        + "Error matrix = " + errstr + "\n"
        + "ERC success matrix = """ + ERCstr + "\n"
        + "Gamma matrix = " + gammastr + "\n")

    def plot(self, subplot=True, solve=True, check=False, thresh=None, show=True, basename=None, saveexts=[], showtitle=False):
        # plt.ion() # Turn interactive off

        if solve is False and check is False:
            RuntimeError('Nothing to plot (both solve and check are False)')

        if show is False and saveexts is None:
            RuntimeError('Neither showing nor saving plot!')

        if basename is None:
            strdatetime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
            basename = []
            if solve is True:
                if subplot is True:
                    basename = basename + [strdatetime + "_err"]
                else:
                    basename = basename + [strdatetime + "_err_" + str(i) for i in range(len(self.solvers))]
            if check is True:
                if subplot is True:
                    basename = basename + [strdatetime + "_erc_"]
                else:
                    basename = basename + [strdatetime + "_erc_" + str(i) for i in range(len(self.ERCsolvers))]
        # if a string, convert to a list
        #if isinstance(basename, types.StringTypes):
        # Python3: use isinstance(s, str)
        if isinstance(basename, str):

            basename = [basename]
        iterFilename = iter(basename)

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
                if showtitle:
                    plt.title(datasources_titles[idatasource][icurrentdata])
                plt.xlabel(r"$\delta$")
                plt.ylabel(r"$\rho$")
                # Show x and y ticks: always 3 ticks: left, middle, right
                tcks = [0, round((self.deltas.size-1)/2), self.deltas.size-1]
                plt.xticks(tcks, ["%.2f"%(val) for val in self.deltas[tcks]])
                tcks = [0, round((self.rhos.size-1)/2), self.rhos.size-1]
                plt.yticks(tcks, ["%.2f"%(val) for val in self.rhos[tcks]])

                if not subplot:
                    # separate figure, save each
                    filename = next(iterFilename)
                    for ext in saveexts:
                        plt.savefig(filename + '.' + ext, bbox_inches='tight')
            if subplot:
                #plt.tight_layout()
                # single figure, save at finish
                filename = next(iterFilename)
                for ext in saveexts:
                    plt.savefig(filename + '.' + ext, bbox_inches='tight')
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

    def savedata(self, basename=None):
        """
        Saves data and parameters to mat file
        :return:
        """
        if basename is None:
            basename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

        # dictionary to save
        mdict = {u'signaldim': self.signaldim, u'dictdim': self.dictdim, u'numdata': self.numdata,
                 u'deltas': self.deltas, u'rhos': self.rhos, 
                 u'snr_db_sparse': self.snr_db_sparse, u'snr_db_signal': self.snr_db_signal, u'snr_db_meas': self.snr_db_meas,
                 u'solverNames': self.solverNames, u'ERCsolverNames': self.ERCsolverNames,
                 u'err': self.err, u'ERCsuccess': self.ERCsuccess, u'gamma': self.gamma, u'support': self.support, u'simData': self.simData,
                 u'description': self.get_description()}

        hdf5storage.savemat(basename + '.mat', mdict)
        with open(basename+"_data.pickle", "wb") as f:
            cPickle.dump(mdict, f)
        with open(basename+".pickle", "wb") as f:
            cPickle.dump(self.solvers, f)
            cPickle.dump(self.ERCsolvers, f)
        with open(basename+".txt", "w") as f:
            f.write(self.get_description())

    def savedescription(self, basename=None):
        """
        Saves description to a text file
        :return:
        """
        if basename is None:
            basename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

        with open(basename+".txt", "w") as f:
            f.write(self.get_description())

    def loaddata(self, matfilename=None, picklefilename1=None, picklefilename2=None):
        """
        Loads data from saved files. If matfilename is not None, all numerical data is read from ithe file (but not
         the solver objects). If picklefilename is not None, solver objects are read from the file.
        :param matfilename:
        :param picklefilename:
        :return:
        """

        if picklefilename1 is not None:
            with open(picklefilename1, "rb") as f:
                solvers = cPickle.load(f)
                self.set_solvers(solvers)
        if matfilename is not None or picklefilename2 is not None:
            if picklefilename2 is not None:
                with open(picklefilename2, "rb") as f:
                    mdict = cPickle.load(f)
            if matfilename is not None:
                mdict = hdf5storage.loadmat(matfilename)
        
            self.signaldim = mdict[u'signaldim']
            self.dictdim = mdict[u'dictdim']
            self.numdata = mdict[u'numdata']
            self.deltas = mdict[u'deltas'].copy()
            self.rhos = mdict[u'rhos'].copy()
            self.snr_db_sparse = mdict[u'snr_db_sparse']
            self.snr_db_signal = mdict[u'snr_db_signal']
            self.snr_db_meas   = mdict[u'snr_db_meas']
            self.solverNames = mdict[u'solverNames'].copy()
            self.ERCsolverNames = mdict[u'ERCsolverNames'].copy()
            if mdict[u'err'] is not None:
                self.err = mdict[u'err'].copy()
            if mdict[u'ERCsuccess'] is not None:
                self.ERCsuccess = mdict[u'ERCsuccess'].copy()
            if mdict[u'gamma'] is not None:
                self.gamma = mdict[u'gamma'].copy()
            if mdict[u'support'] is not None:
                self.support = mdict[u'support'].copy()   # This is a list
            if u'simData' in mdict.keys():
                self.simData = mdict[u'simData']

    @classmethod
    def dump(self, filename):
        """
        Dumps all PhaseTransition object in a file using pickle
        :return: Nothing
        """
        with open(filename, "w") as f:
            cPickle.dump(self,f)

    @classmethod
    def load(cls, filename):
        """
        Classmethod.
        Load complete PhaseTransition object dumped with pickle.dump()
        :return: The loaded object
        """
        with open(filename, "rb") as f:
            obj = cPickle.load(f)
        return obj


    def compute_global_average_error(self, shape, thresh=None, textfilename=None):
        """
        Returns an array same shape as 'solvers' array, containing the average value of the phase transition
        """
        avgerr = self._compute_average(self.err, thresh)
        global_avgerr = np.mean(avgerr, (1,2))
        if textfilename is not None:
            with open(textfilename, "w") as f:
                for solvername, value in zip(self.solverNames, global_avgerr):
                    f.write(solvername + ': ' + str(value) + '\n')

        return np.mean(avgerr, (1,2)).reshape(shape, order='C') # mean over all axes except 0, reshape in row order

    def plot_global_error(self, shape, thresh=None, show=True, basename=None, saveexts=[], textfilename=None, scaling = ["min", "max"]):
        """
        Plots an array same shape as 'solvers' array, containing the average value of the phase transition
        """

        if show is False and saveexts is None and textfilename is None:
            RuntimeError('Neither showing nor saving plot nor writing data!')

        if basename is None:
            basename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")

        plt.figure()
        values = self.compute_global_average_error(shape, thresh, textfilename)

        # Normalize
        minvalue = np.min(values)
        maxvalue = np.max(values)
        if scaling == "percent_max":
            values = values / maxvalue
        else:
            if scaling[0] == "min":
                scaling[0] = minvalue
            if scaling[1] == "max":
                scaling[1] = maxvalue
            values = scaling[0] + (values - minvalue) / (maxvalue - minvalue) * (scaling[1] - scaling[0])

        #values = values - np.min(values)  # translate minimum to 0
        #values = values / np.max(values)  # translate maximum to 1

        if thresh is None:
            plot_phase_transition(values, reverse_colormap = True, transpose=False)
        else:
            plot_phase_transition(values, reverse_colormap = False, transpose=False)

        if show:
            plt.show()

        for ext in saveexts:
            plt.savefig(basename + '.' + ext, bbox_inches='tight')


class SynthesisPhaseTransition(PhaseTransition):
    """
    Class for running and plotting synthesis-based phase transitions
    """

    def __init__(self, signaldim, dictdim, deltas, rhos, numdata, snr_db_sparse, snr_db_signal, snr_db_meas, solvers=[], dictionary="randn", acqumatrix="randn"):
        super(SynthesisPhaseTransition, self).__init__(signaldim, dictdim, deltas, rhos, numdata, snr_db_sparse, snr_db_signal, snr_db_meas, solvers)
        self.dictionary=dictionary
        self.acqumatrix=acqumatrix

    def run(self, solve=True, check=False, processes=None, random_state=None):

        # Both can be False: only generates compressed sensing problems data
        #if solve is False and check is False:
        #    RuntimeError('Nothing to run (both solve and check are False)')

        # Number of processes
        if processes is None:
            processes = multiprocessing.cpu_count()

        # Initialize zero-filled arrays
        if solve is True:
            #self.err = [np.zeros(shape=(len(self.deltas), len(self.rhos), self.numdata)) for _ in self.solvers]
            self.err   = np.zeros(shape=(len(self.solvers), len(self.deltas), len(self.rhos), self.numdata))
            self.gamma = np.zeros(shape=(len(self.solvers), len(self.deltas), len(self.rhos), self.dictionary.shape[1], self.numdata))
            self.support = [[[[] for r in self.rhos] for d in self.deltas] for s in self.solvers]
            

        if check is True:
            #self.ERCsuccess = [np.zeros(shape=(len(self.deltas), len(self.rhos), self.numdata), dtype=bool)
            #                   for _ in self.ERCsolvers]
            self.ERCsuccess = np.zeros(shape=(len(self.ERCsolvers), len(self.deltas), len(self.rhos), self.numdata), dtype=bool)

        if not self.simData:
            # a 2D list of dictionaries, size deltas x rhos
            self.simData = [[dict() for _ in self.rhos] for _ in self.deltas]

        # Generate data if needed
        for idelta, delta in enumerate(self.deltas):
            for irho, rho in enumerate(self.rhos):
                m = int(round(self.signaldim * delta, 0))  # delta = m/n
                k = int(round(m * rho, 0))  # rho = k/m

                if not self.simData[idelta][irho]:
                    measurements, acqumatrix, realdata, dictionary, realgamma, realsupport, cleardata = \
                        gen.make_compressed_sensing_problem(
                            m, self.signaldim, self.dictdim, k, self.numdata, self.snr_db_sparse, self.snr_db_signal, self.snr_db_meas, self.dictionary, self.acqumatrix, random_state=random_state)
                    self.simData[idelta][irho][u'measurements'] = measurements
                    self.simData[idelta][irho][u'acqumatrix'] = acqumatrix
                    self.simData[idelta][irho][u'realdata'] = realdata
                    self.simData[idelta][irho][u'dictionary'] = dictionary
                    self.simData[idelta][irho][u'realgamma'] = realgamma
                    self.simData[idelta][irho][u'realsupport'] = realsupport
                    self.simData[idelta][irho][u'cleardata'] = cleardata

        # Only run if solve or check
        if solve or check:

            # Generate map parameters
            task_parameters = [(self.solvers,
                                self.ERCsolvers,
                                self.simData[idelta][irho][u'measurements'],
                                self.simData[idelta][irho][u'acqumatrix'],
                                self.simData[idelta][irho][u'dictionary'],
                                self.simData[idelta][irho][u'realdata'],
                                self.simData[idelta][irho][u'realgamma'],
                                self.simData[idelta][irho][u'realsupport'],
                                self.simData[idelta][irho][u'cleardata'],
                                solve,
                                check
                               )
                               for idelta in range(len(self.deltas)) for irho in range(len(self.rhos))
                               ]

            # Run tasks, possibly in parallel
            if processes is not 1:
                #if pool is None:
                pool = multiprocessing.Pool(processes=processes)
                results = pool.map(run_synthesis_delta_rho, task_parameters)
            else:
                results = map(run_synthesis_delta_rho, task_parameters)


            # Process results
            result_iter = iter(results)
            for idelta in range(len(self.deltas)):
                for irho in range(len(self.rhos)):
                    result = next(result_iter)

                    # Unpack results
                    res_err        = result[0]
                    res_ERCsuccess = result[1]
                    res_gamma      = result[2]
                    res_supp       = result[3]

                    if solve is True:
                        self.err[:,idelta,irho,:] = res_err
                        self.gamma[:,idelta,irho, :, :] = res_gamma
                        for isolver, res_supp_solver in enumerate(res_supp):
                            self.support[isolver][idelta][irho]  = res_supp_solver

                    if check is True:
                        self.ERCsuccess[:,idelta,irho,:] = res_ERCsuccess



def run_synthesis_delta_rho(tuple_data):

    # Unpack tuple
    solvers = tuple_data[0]
    ERCsolvers = tuple_data[1]
    measurements = tuple_data[2]
    acqumatrix = tuple_data[3]
    dictionary = tuple_data[4]
    realdata = tuple_data[5]
    realgamma = tuple_data[6]
    realsupport = tuple_data[7]
    cleardata = tuple_data[8]
    solve = tuple_data[9]
    check = tuple_data[10]

    realdict = {'data': realdata, 'gamma': realgamma, 'support': realsupport}

    # Prepare results
    num_data = measurements.shape[1]
    ERCsuccess = np.zeros(shape=(len(ERCsolvers), num_data), dtype=bool)
    err = np.zeros(shape=(len(solvers), num_data))
    gammaout = np.zeros(shape=(len(solvers), dictionary.shape[1], num_data))
    suppout = []  # pass a list not an array the support of each signal may have different lengths

    if check is True:
        for iERCsolver, ERCsolver in enumerate(ERCsolvers):
            #self.ERCsuccess[iERCsolver, idelta, irho] = ERCsolver.checkERC(acqumatrix, dictionary, realsupport)
            ERCsuccess[iERCsolver] = ERCsolver.checkERC(acqumatrix, dictionary, realsupport)
    if solve is True:
        for isolver, solver in enumerate(solvers):
            result = solver.solve(measurements, np.dot(acqumatrix, dictionary), realdict)
            
            # Support solvers which return a tuple (gamma, support list) as well as the older ones which return only gamma
            if isinstance(result, tuple):
                gamma = result[0]
                supp = result[1]
            else:
                gamma = result
                # Attempt to discover support based on non-zero coefficients in solution
                supp = []
                for isig in range(gamma.shape[1]):
                    supp.append(np.array(np.nonzero(gamma[:,isig])[0]))
           
            # Compute and save relative error
            data = np.dot(dictionary, gamma)
            errors = data - realdata
            for i in range(errors.shape[1]):
                errors[:, i] = errors[:, i] / np.linalg.norm(realdata[:, i])
                err[isolver][i] = np.sqrt(sum(errors[:, i] ** 2))

            # Save gamma for output
            gammaout[isolver] = gamma

            # Save support for output
            suppout.append(supp)

    return err, ERCsuccess, gammaout, suppout


# TODO: maybe needs refactoring, it's very similar to PhaseTransition()
class AnalysisPhaseTransition(PhaseTransition):
    """
    Class for running and plotting analysis-based phase transitions
    """

    def __init__(self, signaldim, operatordim, deltas, rhos, numdata, snr_db, solvers=[], oper_type="randn", acqu_type="randn"):
        super(AnalysisPhaseTransition, self).__init__(signaldim, operatordim, deltas, rhos, numdata, snr_db, solvers)
        self.oper_type=oper_type
        self.acqu_type=acqu_type

    def run(self, solve=True, check=False, processes=None, random_state=None):

        # Number of processes
        if processes is None:
            processes = multiprocessing.cpu_count()
        pool = None

        # Initialize zero-filled arrays
        if solve is True:
            self.err = np.zeros(shape=(len(self.solvers), len(self.deltas), len(self.rhos), self.numdata))
        if check is True:
            self.ERCsuccess = \
                np.zeros(shape=(len(self.ERCsolvers), len(self.deltas), len(self.rhos), self.numdata), dtype=bool)

        if not self.simData:
            self.simData = [[dict() for _ in self.rhos] for _ in self.deltas]

        # Generate data if needed
        # for idelta, delta in enumerate(self.deltas):
        #     for irho, rho in enumerate(self.rhos):
        #         m = int(round(self.signaldim * delta, 0))  # delta = m/n
        #         l = self.signaldim - int(round(m * rho, 0))  # rho = (n-l)/m
        #
        #         if not self.simData[idelta][irho]:
        #             measurements, acqumatrix, realdata, operator, realgamma, realcosupport, cleardata = \
        #                 gen.make_analysis_compressed_sensing_problem(
        #                     m, self.signaldim, self.dictdim, l, self.numdata, self.snr_db, operator=self.oper_type, acquisition=self.acqu_type)
        #             self.simData[idelta][irho][u'measurements'] = measurements
        #             self.simData[idelta][irho][u'acqumatrix'] = acqumatrix
        #             self.simData[idelta][irho][u'realdata'] = realdata
        #             self.simData[idelta][irho][u'operator'] = operator
        #             self.simData[idelta][irho][u'realgamma'] = realgamma
        #             self.simData[idelta][irho][u'realcosupport'] = realcosupport
        #             self.simData[idelta][irho][u'cleardata'] = cleardata

        # When multiprocessing, don't use random_state
        if processes is not 1:
            gen_parameters = [(int(round(self.signaldim * delta, 0)), # this is m,  delta = m/n
                                self.signaldim,
                                self.dictdim,
                                self.signaldim - int(round(
                                    int(round(self.signaldim * delta, 0))  # this is m
                                    * rho, 0)), # this is l,  rho = (n-l)/m
                                self.numdata,
                                self.snr_db,
                                self.oper_type,
                                self.acqu_type,
                                None, # random_state
                               )
                               for idelta, delta in enumerate(self.deltas)
                               for irho, rho in enumerate(self.rhos)
                               if not self.simData[idelta][irho]
            ]
        else:
            # add random_state to parameters
            gen_parameters = [(int(round(self.signaldim * delta, 0)), # this is m,  delta = m/n
                               self.signaldim,
                               self.dictdim,
                               self.signaldim - int(round(
                                   int(round(self.signaldim * delta, 0))  # this is m
                                   * rho, 0)), # this is l,  rho = (n-l)/m
                               self.numdata,
                               self.snr_db,
                               self.oper_type,
                               self.acqu_type,
                               random_state,
                              )
                              for idelta, delta in enumerate(self.deltas)
                              for irho, rho in enumerate(self.rhos)
                              if not self.simData[idelta][irho]
            ]

        # Run generation tasks
        if processes is not 1:
            if pool is None:
                pool = multiprocessing.Pool(processes=processes)
            results = pool.map(tuplewrap_make_analysis_compressed_sensing_problem, gen_parameters)
        else:
            results = map(tuplewrap_make_analysis_compressed_sensing_problem, gen_parameters)

        # Process generation results
        result_iter = iter(results)
        for idelta in range(len(self.deltas)):
            for irho in range(len(self.rhos)):
                result = next(result_iter)
                self.simData[idelta][irho][u'measurements'] = result[0]
                self.simData[idelta][irho][u'acqumatrix'] = result[1]
                self.simData[idelta][irho][u'realdata'] = result[2]
                self.simData[idelta][irho][u'operator'] = result[3]
                self.simData[idelta][irho][u'realgamma'] = result[4]
                self.simData[idelta][irho][u'realcosupport'] = result[5]
                self.simData[idelta][irho][u'cleardata'] = result[6]

        # Only run if solve or check
        if solve or check:

            # Make run parameters
            run_parameters = [(self.solvers,
                                self.ERCsolvers,
                                self.simData[idelta][irho][u'measurements'],
                                self.simData[idelta][irho][u'acqumatrix'],
                                self.simData[idelta][irho][u'operator'],
                                self.simData[idelta][irho][u'realdata'],
                                self.simData[idelta][irho][u'realgamma'],
                                self.simData[idelta][irho][u'realcosupport'],
                                self.simData[idelta][irho][u'cleardata'],
                                solve,
                                check
                               )
                               for idelta in range(len(self.deltas)) for irho in range(len(self.rhos))
            ]

            print("Starting solver processes:")
            time_start = datetime.datetime.now()
            print(time_start.strftime("%Y-%m-%d --- %H:%M:%S:%f"))

            # Run run tasks
            if processes is not 1:
                if pool is None:
                    pool = multiprocessing.Pool(processes=processes)
                results = pool.map(run_analysis_delta_rho, run_parameters)
            else:
                results = map(run_analysis_delta_rho, run_parameters)

            # Process results
            result_iter = iter(results)
            for idelta in range(len(self.deltas)):
                for irho in range(len(self.rhos)):
                    result = next(result_iter)
                    if solve is True:
                        self.err[:,idelta,irho,:] = result[0]
                    if check is True:
                        self.ERCsuccess[:,idelta,irho,:] = result[1]

            time_end = datetime.datetime.now()
            print("End time: " + time_end.strftime("%Y-%m-%d --- %H:%M:%S:%f"))
            print("Elapsed: " + str((time_end - time_start).seconds) + " seconds")



# this can be avoided in python 3.3
def tuplewrap_make_analysis_compressed_sensing_problem(tuple_data):
    return gen.make_analysis_compressed_sensing_problem(*tuple_data)

def run_analysis_delta_rho(tuple_data):

    # Unpack tuple
    solvers = tuple_data[0]
    ERCsolvers = tuple_data[1]
    measurements = tuple_data[2]
    acqumatrix = tuple_data[3]
    operator = tuple_data[4]
    realdata = tuple_data[5]
    realgamma = tuple_data[6]
    realcosupport = tuple_data[7]
    cleardata = tuple_data[8]
    solve = tuple_data[9]
    check = tuple_data[10]

    realdict = {'data': realdata, 'gamma': realgamma, 'cosupport': realcosupport}

    realsupport = np.zeros((operator.shape[0] - realcosupport.shape[0], realcosupport.shape[1]), dtype=int)
    for i in range(realcosupport.shape[1]):
        realsupport[:, i] = np.setdiff1d(range(operator.shape[0]), realcosupport[:, i])

    # Prepare results
    num_data = measurements.shape[1]
    ERCsuccess = np.zeros(shape=(len(ERCsolvers), num_data), dtype=bool)
    err = np.zeros(shape=(len(solvers), num_data))

    if check is True:
        for iERCsolver, ERCsolver in enumerate(ERCsolvers):
            ERCsuccess[iERCsolver] = ERCsolver.checkERC(acqumatrix, operator, realsupport)
    if solve is True:
        for isolver, solver in enumerate(solvers):
            data = solver.solve(measurements, acqumatrix, operator, realdict)
            errors = data - realdata
            for i in range(errors.shape[1]):
                errors[:, i] = errors[:, i] / np.linalg.norm(realdata[:, i])
                err[isolver,i] = np.sqrt(sum(errors[:, i] ** 2))

    return err, ERCsuccess


# TODO: add many more parameters
def plot_phase_transition(matrix, transpose=True, reverse_colormap=False, xvals=[], yvals=[]):
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

    if reverse_colormap:
        cmap = cm.gray_r
    else:
        cmap = cm.gray
    plt.imshow(bigmatrix, cmap=cmap, norm=mcolors.Normalize(0, 1), interpolation='nearest', origin='lower')
    if xvals:
        plt.xticks(xvals)
    if yvals:
        plt.yticks(yvals)


#====================================================

class SparseCodingMixin:

    def __init__(self, *args):
        self.err = None
        self.deltas = None
        self.rhos = None
        self.solverNames = None
        super().__init__(*args)  # mixin calls super too


    def plot(self, thresh=None, basename=None, saveexts=[], showtitle=False, legend=[], rhomax=1, plot_options={}, 
                xlim=(None, None), ylim=(None, None)):

        markers = ['o','x','^','v','+']

        if basename is None:
            strdatetime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
            basename = 'plot_lines_' + strdatetime

        datasources = []
        reverse_colormap = []


        if self.err is None:
            ValueError("No data to plot (have you run()?)")
        else:
            datasources.append(self._compute_average(self.err, thresh))
            if thresh is None:
                reverse_colormap.append(True)
            else:
                reverse_colormap.append(False)

        for data in datasources:

            fig = plt.figure()

            if plot_options:
                color_list       = [plot_options[algo]['color'] for algo in legend]
                marker_list      = [plot_options[algo]['marker'] for algo in legend]
                linestyle_list   = [plot_options[algo]['linestyle'] for algo in legend]

            else:
                # Use default colors, set markers
                color_list     = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
                #marker_list=['.','x', '+', 's','D','^','V', '<', '>'])                
                marker_list    = ['o','^','v', '<', '>', 's', 'D', '*', 'X', 'P']
                linestyle_list = ['-']*len(color_list)

            # Set styles
            plt.gca().set_prop_cycle(color=color_list,
                                     marker=marker_list,
                                     linestyle=linestyle_list)
            
            # Find index of first rho value larger than rhomax, or len(self.rhos) if no value is larger (i.e. keep all rhos)
            # Will plot data only up to irho_limit on the x axis
            irho_limit = next((idx for idx, value in enumerate(self.rhos) if value > rhomax), len(self.rhos)) 

            plt.plot(self.rhos[:irho_limit], data[:,0,:irho_limit].T)   # data[algorithm, idelta, irho].Transposed => [irho, algo]

            # Set axis limits
            x1, x2 = xlim
            y1, y2 = ylim
            if x1 is not None: plt.xlim(left=x1)
            if x2 is not None: plt.xlim(right=x1)
            if y1 is not None: plt.ylim(bottom=y1)
            if y2 is not None: plt.ylim(top=y2)

            plt.xlabel(r"Relative sparsity $\rho$ = k / n")
            if thresh is None:
                plt.ylabel(r"Average recovery error")
            else:
                plt.ylabel(r"Fraction of successfully recovered signals")

            if showtitle:
                if thresh is None:
                    plt.title('Average recovery error depending on sparsity level')
                else:
                    plt.title('Fraction of successfully recovered signals depending on sparsity level')
            
            if legend:
                plt.legend(legend)

            for ext in saveexts:
                if ext != 'pickle':
                    plt.savefig(basename + '.' + ext, bbox_inches='tight')
                else:
                    cPickle.dump(fig, open(basename + '.pickle','wb'))

            plt.close()

    
    def plot_suppport_recovered(self, basename=None, saveexts=[], showtitle=False, legend=[], rhomax=1):

        markers = ['o','x','^','v','+']

        if basename is None:
            strdatetime = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
            basename = 'plot_lines_' + strdatetime

        if self.support is None:
            ValueError("No support information available")

        # Compute percent of recovered support
        percent_good_support = np.zeros(shape=(len(self.solvers), len(self.deltas), len(self.rhos), self.numdata))
        for idelta, _ in enumerate(self.deltas):
            for irho, _ in enumerate(self.rhos):
                realsupp = self.simData[idelta][irho][u'realsupport']
                for isolver,_ in enumerate(self.solvers):
                    for isig, recsupp in enumerate(self.support[isolver][idelta][irho]):     # support for each signal
                        percent_good_support[isolver, idelta, irho, isig] = len(set(realsupp[:,isig]) & set(recsupp)) / len(realsupp[:,isig])

        datasources = [percent_good_support.mean(axis=3)]   # Average over last axis (signals)

        for data in datasources:

            fig = plt.figure()

            # Set styles
            # Use default colors, set markers
            plt.gca().set_prop_cycle(color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'],
                                    #marker=['.','x', '+', 's','D','^','V', '<', '>'])
                                    marker=['o','^','v', '<', '>', 's', 'D', '*', 'X', 'P'])
            
            # Find index of first rho value larger than rhomax, or len(self.rhos) if no value is larger (i.e. keep all rhos)
            # Will plot data only up to irho_limit on the x axis
            irho_limit = next((idx for idx, value in enumerate(self.rhos) if value > rhomax), len(self.rhos)) 

            plt.plot(self.rhos[:irho_limit], data[:,0,:irho_limit].T)   # data[algorithm, idelta, irho].Transposed => [irho, algo]

            plt.xlabel(r"Relative sparsity $\rho$ = k / n")
            plt.ylabel(r"Fraction of successfully recovered support")

            if showtitle:
                plt.title('Fraction of successfully recovered support depending on sparsity level')
            
            if legend:
                plt.legend(legend)

            for ext in saveexts:
                if ext != 'pickle':
                    plt.savefig(basename + '.' + ext, bbox_inches='tight')
                else:
                    cPickle.dump(fig, open(basename + '.pickle','wb'))

            plt.close()

class SynthesisSparseCoding(SparseCodingMixin, SynthesisPhaseTransition):
    def __init__(self, signaldim=None, dictdim=None, rhos=[], ks=None, numdata=1, snr_db_sparse=np.Inf, snr_db_signal=np.Inf, solvers=[], dictionary="randn"):

        if signaldim == None:
            signaldim = dictionary.shape[0]
        if dictdim == None:
            dictdim = dictionary.shape[1]
        if rhos == []:
            rhos = [k / signaldim for k in ks] 
        deltas = [signaldim / dictdim]

        # Phase Transition maps to Sparse Coding:
        #  measurement matrix = dictionary
        #  dictionary = unit matrix
        #  measurements = the signal
        #  the signal = the sparse decomposition
        #  the sparse decomposition = the sparse decomposition (dictionary = unit matrix so signal = decomposition)
        #  measurement noise = signal noise
        #  signal noise = decompostion noise
        #  decomposition noise = another decompostion noise

        super().__init__(dictdim, dictdim, deltas, rhos, numdata, np.Inf, snr_db_sparse, snr_db_signal, solvers, np.eye(dictdim), dictionary) # dictionary is eye(), acqumatrix is dictionary