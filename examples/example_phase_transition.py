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


def run_synthesis():
    signal_size, dict_size = 50, 70
    deltas = numpy.arange(0.1, 0.9, 0.1)
    rhos = numpy.arange(0.1, 0.9, 0.1)

    print "Running synthesis phase transition..."
    pt = SynthesisPhaseTransition(signal_size, dict_size, deltas, rhos, 3,
                                  [OrthogonalMatchingPursuit(1e-6, algorithm="sparsify_QR"),
                                   L1Min(1e-6),
                                   SmoothedL0(1e-6),
                                   TwoStageThresholding(1e-6)])
    pt.run()
    pt.plot()

    print "Example finished."


def run_analysis():
    signal_size, dict_size = 50, 70
    deltas = numpy.arange(0.1, 0.9, 0.1)
    rhos = numpy.arange(0.1, 0.9, 0.1)

    print "Running analysis phase transition..."
    pt = AnalysisPhaseTransition(signal_size, dict_size, deltas, rhos, 3,
                                 [AnalysisL1Min(1e-6),
                                  AnalysisBySynthesis(L1Min(1e-6)),
                                  AnalysisBySynthesis(OrthogonalMatchingPursuit(1e-6, algorithm="sparsify_QR")),
                                  AnalysisBySynthesis(SmoothedL0(1e-6)),
                                  GreedyAnalysisPursuit(1e-6)])
    pt.run()
    pt.plot()

def run_uap():
    signal_size, dict_size = 50, 70
    deltas = numpy.arange(0.1, 0.9, 0.1)
    rhos = numpy.arange(0.1, 0.9, 0.1)

    print "Running analysis phase transition..."
    pt = AnalysisPhaseTransition(signal_size, dict_size, deltas, rhos, 3,
                                 [UnconstrainedAnalysisPursuit(1e-6, 1, 1),
                                  GreedyAnalysisPursuit(1e-6)])
    pt.run()
    pt.plot()


    print "Example finished."


if __name__ == "__main__":
    #run_synthesis()
    #run_analysis()
    run_uap()