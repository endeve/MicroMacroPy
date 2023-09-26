# MicroMacroPy
Python code for solving the micro-macro decomposition of the Vlasov-Poisson-Lenard-Bernstein system.

This code was used to generate the results presented in Endeve &amp; Hauck (2022), JCP, 462, 111227.

## Basic Code Description

All python source code is located under the _Source_ directory.

Drivers for the __Relaxation__, __Riemann__, and __Collisional Landau Damping__ problems are named _RelaxationMM.py_, _RiemannProblemMM.py_, and _CollisionalLandauDampingMM.py_, respectively.
Corresponding drivers using the direct method (solving for the kinetic distribution $f$) are named _Relaxation.py_, _RiemannProblem.py_, and _CollisonalLandauDamping.py_.  

## Running the Code