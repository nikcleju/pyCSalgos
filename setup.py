from distutils.core import setup
setup(
    name = "pyCSalgos",
    packages = ["pyCSalgos","pyCSalgos/ABS","pyCSalgos/BP","pyCSalgos/GAP","pyCSalgos/NESTA","pyCSalgos/OMP","pyCSalgos/TST","pyCSalgos/SL0"],
    version = "1.1.0",
    description = "Python Compressed Sensing algorithms",
    author = "Nicolae Cleju",
    author_email = "nikcleju@gmail.com",
    url = 'https://code.soundsoftware.ac.uk/projects/pycsalgos',
    classifiers = [
        "Programming Language :: Python",
        "Development Status :: 4 - Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics"
        ],
    long_description = """\
Python Compressed Sensing algorithms
-------------------------------------

Python implementation of various Compressed Sensing algorithms, some of them originally implemented in Matlab.
Algorithms implemented:
- l1 minimization from l1magic
- Orthogonal Matching Pursuit
- Smoothed L0
- Greedy Analysis Pursuit
- NESTA
- Two Stage Thresholding
- Analysis-By-Synthesis (my paper)

Not thoroughly tested, but I use them for my research. Use at own risk. 
"""
)
