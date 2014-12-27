"""
example_phase_transition.py

An example of generating a phase transition plot
"""

import numpy
import datetime
from pyCSalgos import AnalysisPhaseTransition, SynthesisPhaseTransition
from pyCSalgos import GreedyAnalysisPursuit

signal_size, dict_size = 200, 240
deltas = numpy.arange(0.1, 1, 0.1)
rhos = numpy.arange(0.1, 1, 0.1)
operator_type = "tightframe"
num_data = 10

partname = 'GAP'
solver = GreedyAnalysisPursuit(1e-10)

def run_once():

    time_start = datetime.datetime.now()
    print "Start time: " + time_start.strftime("%Y-%m-%d --- %H:%M:%S:%f")

    filebasename = 'save/exact_'+ partname

    pt = AnalysisPhaseTransition(signal_size, dict_size, deltas, rhos, num_data, numpy.inf, [solver], oper_type=operator_type)

    pt.run()
    pt.savedata(filebasename)
    pt.plot(thresh=1e-6, show=False, basename=filebasename, saveexts=['pdf', 'png'])

    time_end = datetime.datetime.now()
    print "End time:   " + time_end.strftime("%Y-%m-%d --- %H:%M:%S:%f")
    print "Elapsed:    " + str((time_end - time_start).seconds) + " seconds"
    print "------"


if __name__ == "__main__":
    print "Running analysis phase transition..."

    run_once()

    print "Finished."