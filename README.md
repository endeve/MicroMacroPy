# MicroMacroPy
Python code for solving the micro-macro decomposition of the Vlasov-Poisson-Lenard-Bernstein system.
This code was used to generate the results presented in Endeve &amp; Hauck (2022), JCP, 462, 111227.

## Basic Code Description

All python source code is located under the _Source_ directory.

Drivers for the __Relaxation__, Riemann, and Collisional Landau Damping problems are named RelaxationMM.py, RiemannProblemMM.py, and CollisionalLandauDampingMM.py, respectively.
Corresponding drivers using the direct method (solving for the kinetic distribution $f$) are named Relaxation.py, RiemannProblem.py, and CollisonalLandauDamping.py.  

## Running the Code