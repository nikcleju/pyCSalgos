from pyCSalgos import make_analysis_compressed_sensing_problem

import cProfile

if __name__ == '__main__':
    cProfile.run(
        'make_analysis_compressed_sensing_problem(10, 50, 60, 47, 100, operator="randn", acquisition="randn")',
        'profile_make_analysis_compressed_sensing_problem_4.prf')
