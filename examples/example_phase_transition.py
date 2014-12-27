"""
example_phase_transition.py

An example of generating a phase transition plot
"""

import numpy
from pyCSalgos import SynthesisPhaseTransition
from pyCSalgos import AnalysisPhaseTransition
from pyCSalgos import OrthogonalMatchingPursuit
from pyCSalgos import L1Min
from pyCSalgos import SmoothedL0
from pyCSalgos import TwoStageThresholding
from pyCSalgos import AnalysisL1Min
from pyCSalgos import AnalysisBySynthesis
from pyCSalgos import GreedyAnalysisPursuit
from pyCSalgos import UnconstrainedAnalysisPursuit
from pyCSalgos import ApproximateMessagePassing
from pyCSalgos import IterativeHardThresholding

def run_test():
    signal_size, dict_size = 200, 240
    deltas = numpy.arange(0.1, 1, 0.1)
    rhos = numpy.arange(0.1, 1, 0.1)

    print "Running synthesis phase transition..."
    pt = SynthesisPhaseTransition(signal_size, dict_size, deltas, rhos, 3, numpy.inf,
                                  [
                                  #OrthogonalMatchingPursuit(0, algorithm="sparsify_QR"),
                                  #IterativeHardThresholding(1e-10, sparsity="real", maxiter=1000)
                                  #ApproximateMessagePassing(1e-6, 1000),
                                  ])
    pt.run(solve=True, check=False, processes=None)
    pt.plot(solve=True, check=False)

    pt.plot(solve=True, check=False, thresh=1e-6)

    #pt.savedata()

    print "Example finished."

def run_synthesis():
    signal_size, dict_size = 50, 100
    deltas = numpy.arange(0.1, 1, 0.03)
    rhos = numpy.arange(0.1, 1, 0.03)
    #signal_size, dict_size = 512,512
    #deltas = [0.5]
    #rhos = numpy.arange(0.1, 0.51, 0.01)


    print "Running synthesis phase transition..."
    pt = SynthesisPhaseTransition(signal_size, dict_size, deltas, rhos, 30,
                                  [OrthogonalMatchingPursuit(1e-6, algorithm="sparsify_QR"),
                                   L1Min(1e-6),
                                   SmoothedL0(1e-6),
                                   TwoStageThresholding(1e-6),
                                   ApproximateMessagePassing(1e-6, 1000),
                                   IterativeHardThresholding(1e-10, sparsity="real", maxiter=1000) ])
    pt.run()
    pt.plot()

    print "Example finished."


def run_analysis():
    signal_size, dict_size = 50, 60
    deltas = numpy.arange(0.1, 0.99, 0.1)
    rhos = numpy.arange(0.1, 0.9, 0.1)

    print "Running analysis phase transition..."
    pt = AnalysisPhaseTransition(signal_size, dict_size, deltas, rhos, 3, numpy.inf,
                                 [AnalysisL1Min(1e-6),
                                  AnalysisBySynthesis(L1Min(1e-6)),
                                  AnalysisBySynthesis(OrthogonalMatchingPursuit(1e-10, algorithm="sparsify_QR")),
                                  AnalysisBySynthesis(SmoothedL0(1e-8)),
                                  GreedyAnalysisPursuit(1e-6)])
    pt.run(processes=1)
    pt.plot(thresh=1e-6)

def run_uap():
    signal_size, dict_size = 50, 70
    deltas = numpy.arange(0.1, 1, 0.1)
    rhos = numpy.arange(0.1, 1, 0.1)

    print "Running analysis phase transition..."
    pt = AnalysisPhaseTransition(signal_size, dict_size, deltas, rhos, 3,
                                 [UnconstrainedAnalysisPursuit(1e-6, 1, 1),
                                  GreedyAnalysisPursuit(1e-6)])
    pt.run()
    pt.plot()


    print "Example finished."


if __name__ == "__main__":

    # Profile
    #import profile
    #profile.run('run_test()', 'profile.tmp')

    #import pstats
    #p = pstats.Stats('profile.tmp')
    #p.sort_stats('cumulative').print_stats(10)

    #run_synthesis()
    run_analysis()
    #run_uap()
    #run_test()