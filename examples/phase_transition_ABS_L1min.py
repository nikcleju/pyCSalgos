"""
example_phase_transition.py

An example of generating a phase transition plot
"""

import numpy
import datetime
from pyCSalgos import AnalysisPhaseTransition
from pyCSalgos import L1Min
from pyCSalgos import AnalysisBySynthesis

signal_size, dict_size = 200, 240
deltas = numpy.arange(0.1, 1, 0.1)
rhos = numpy.arange(0.1, 1, 0.1)
operator_type = "tightframe"
num_data = 10

partname = 'ABS_L1MIN'
synth_solver = L1Min(0, algorithm="cvxopt")

def run_once():

    time_start = datetime.datetime.now()
    print "Start time: " + time_start.strftime("%Y-%m-%d --- %H:%M:%S:%f")

    solver = AnalysisBySynthesis(synth_solver)

    pt = AnalysisPhaseTransition(signal_size, dict_size, deltas, rhos, num_data, numpy.inf, [solver], oper_type=operator_type)

    filebasename = 'save/exact _'+ partname
    pt.run()
    pt.savedata(filebasename)

    pt.plot(thresh=1e-6, show=False, basename=filebasename, saveexts=['pdf', 'png'])

    time_end = datetime.datetime.now()
    print "End time:   " + time_end.strftime("%Y-%m-%d --- %H:%M:%S:%f")
    print "Elapsed:    " + str((time_end - time_start).seconds) + " seconds"
    print "------"

def run_lambda_vals():

    time_start = datetime.datetime.now()
    print "Start time: " + time_start.strftime("%Y-%m-%d --- %H:%M:%S:%f")

    lambdas = numpy.logspace(-4,4,9)
    solvers = [AnalysisBySynthesis(synth_solver,
                                   nullspace_multiplier=lmbd) for lmbd in lambdas]

    pt = AnalysisPhaseTransition(signal_size, dict_size, deltas, rhos, num_data, numpy.inf, solvers, oper_type=operator_type)

    filebasename = 'save/exact_'+ partname + '_lambdavals'
    pt.run()
    pt.savedata(filebasename)

    pt.plot(thresh=1e-6, subplot=True, show=False, basename=filebasename, saveexts=['pdf', 'png'])
    pt.plot_global_error(shape=(1,len(solvers)), thresh=1e-6, show=False, basename=filebasename+'_globalerr', saveexts=['pdf', 'png'])

    time_end = datetime.datetime.now()
    print "End time:   " + time_end.strftime("%Y-%m-%d --- %H:%M:%S:%f")
    print "Elapsed:    " + str((time_end - time_start).seconds) + " seconds"
    print "------"

def run_lambda_type():

    time_start = datetime.datetime.now()
    print "Start time: " + time_start.strftime("%Y-%m-%d --- %H:%M:%S:%f")

    types = ["value", "normalized_row"]
    solvers = [AnalysisBySynthesis(synth_solver,
                                   nullspace_multiplier=1,
                                   nullspace_multiplier_type=type) for type in types]

    pt = AnalysisPhaseTransition(signal_size, dict_size, deltas, rhos, num_data, numpy.inf, solvers, oper_type=operator_type)

    filebasename = 'save/exact_'+ partname + '_lambdatype'
    pt.run()
    pt.savedata(filebasename)

    pt.plot(thresh=1e-6, subplot=True, show=False, basename=filebasename, saveexts=['pdf', 'png'])
    pt.plot_global_error(shape=(1,len(solvers)), thresh=1e-6, show=False, basename=filebasename+'_globalerr', saveexts=['pdf', 'png'])

    time_end = datetime.datetime.now()
    print "End time:   " + time_end.strftime("%Y-%m-%d --- %H:%M:%S:%f")
    print "Elapsed:    " + str((time_end - time_start).seconds) + " seconds"
    print "------"

def run_lambda_vals_normrow():

    time_start = datetime.datetime.now()
    print "Start time: " + time_start.strftime("%Y-%m-%d --- %H:%M:%S:%f")

    lambdas = numpy.logspace(-4,4,9)
    solvers = [AnalysisBySynthesis(synth_solver,
                                   nullspace_multiplier=lmbd,
                                   nullspace_multiplier_type="normalized_row") for lmbd in lambdas]

    pt = AnalysisPhaseTransition(signal_size, dict_size, deltas, rhos, num_data, numpy.inf, solvers, oper_type=operator_type)

    filebasename = 'save/exact_'+ partname + '_lambdavals_normrow'
    pt.run()
    pt.savedata(filebasename)

    pt.plot(thresh=1e-6, subplot=True, show=False, basename=filebasename, saveexts=['pdf', 'png'])
    pt.plot_global_error(shape=(1,len(solvers)), thresh=1e-6, show=False, basename=filebasename+'_globalerr', saveexts=['pdf', 'png'])

    time_end = datetime.datetime.now()
    print "End time:   " + time_end.strftime("%Y-%m-%d --- %H:%M:%S:%f")
    print "Elapsed:    " + str((time_end - time_start).seconds) + " seconds"
    print "------"

if __name__ == "__main__":
    print "Running analysis phase transition..."

    run_once()
    run_lambda_vals()
    run_lambda_type()
    run_lambda_vals_normrow()


print "Finished."