# MicroMacroPy
Python code for solving the micro-macro decomposition of the Vlasov-Poisson-Lenard-Bernstein system.

This code was used to generate the results presented in Endeve &amp; Hauck (2022), JCP, 462, 111227.

In the micro-macro decomposition, the kinetic distribution function is decomposed as $f=M_{f}+g$, where $M_{f}$ is the Maxwellian with the same first three velocity moments as $f$; i.e., $\rho_{f}=\langle M_{f} \mathbf{e} \rangle=\langle f \mathbf{e} \rangle$

## Basic Code Description

All python source code is located under the _Source_ directory.

Drivers for the __Relaxation__, __Riemann__, and __Collisional Landau Damping__ problems are named _RelaxationMM.py_, _RiemannProblemMM.py_, and _CollisionalLandauDampingMM.py_, respectively.
Corresponding drivers using the direct method (solving for the kinetic distribution $f$) are named _Relaxation.py_, _RiemannProblem.py_, and _CollisonalLandauDamping.py_.  

## Running the Code