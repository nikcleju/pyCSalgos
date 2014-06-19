"""
example_phase_transition.py

An example of generating a phase transition plot
"""

import numpy
from pyCSalgos.phase_transition import PhaseTransition
from pyCSalgos.omp import OrthogonalMatchingPursuit
from pyCSalgos.l1min import L1Min

def run():
    n, N = 50, 70
    deltas = numpy.arange(0.1, 0.9, 0.1)
    rhos = numpy.arange(0.1, 0.9, 0.1)
    pt = PhaseTransition(n, N, deltas, rhos, 3, [OrthogonalMatchingPursuit(1e-6, algorithm="sparsify_QR"), L1Min(1e-6)])
    pt.run()
    pt.plot()
    print "Example finished."

if __name__ == "__main__":
    run()