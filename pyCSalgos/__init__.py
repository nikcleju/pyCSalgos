"""
The :mod:`pyCSalgos` module: Python Compressed Sensing algorithms.
"""

from .generate import make_sparse_coded_signal
from .generate import make_compressed_sensing_problem
from .generate import make_cosparse_coded_signal
from .generate import make_analysis_compressed_sensing_problem

from .omp import OrthogonalMatchingPursuit
from .l1min import L1Min
from .sl0 import SmoothedL0
from .tst import TwoStageThresholding
from .amp import ApproximateMessagePassing
from .iht import IterativeHardThresholding

from .analysisl1min import AnalysisL1Min
from .gap import GreedyAnalysisPursuit
from .analysis_by_synthesis import AnalysisBySynthesis
from .uap import UnconstrainedAnalysisPursuit

from .phase_transition import SynthesisPhaseTransition
from .phase_transition import AnalysisPhaseTransition


__all__ = ['make_sparse_coded_signal',
           'make_compressed_sensing_problem',
           'make_cosparse_coded_signal',
           'make_analysis_compressed_sensing_problem',
           'OrthogonalMatchingPursuit',
           'L1Min',
           'SmoothedL0',
           'TwoStageThresholding',
           'ApproximateMessagePassing',
           'IterativeHardThresholding',
           'AnalysisL1Min',
           'GreedyAnalysisPursuit',
           'AnalysisBySynthesis',
           'UnconstrainedAnalysisPursuit',
           'SynthesisPhaseTransition',
           'AnalysisPhaseTransition']