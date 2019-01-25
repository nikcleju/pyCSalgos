import cProfile

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


def run():
    signal_size, dict_size = 200, 240
    deltas = numpy.arange(0.1, 1, 0.3)
    rhos = numpy.arange(0.1, 1, 0.3)

    print("Running analysis phase transition...")
    pt = AnalysisPhaseTransition(signal_size, dict_size, deltas, rhos, 3,
                                 [GreedyAnalysisPursuit(1e-8),
                                  #AnalysisL1Min(1e-8),
                                  #AnalysisBySynthesis(L1Min(1e-6)),
                                  #AnalysisBySynthesis(OrthogonalMatchingPursuit(1e-9, algorithm="sparsify_QR")),
                                  #AnalysisBySynthesis(SmoothedL0(1e-8)),
                                  #AnalysisBySynthesis(TwoStageThresholding(1e-8, maxiter=3000)),
                                  #AnalysisBySynthesis(IterativeHardThresholding(1e-10, sparsity="real", maxiter=1000))
                                 ])
    pt.run()
    #pt.plot(thresh=1e-6)
    #pt.save()

if __name__ == '__main__':
    cProfile.run('run()',
        'profile_GAP')
