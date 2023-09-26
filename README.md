# MicroMacroPy
Python code for solving the micro-macro decomposition of the Vlasov-Poisson-Lenard-Bernstein system.

This code was used to generate the results presented in Endeve &amp; Hauck (2022), JCP, 462, 111227.

In the micro-macro decomposition, the kinetic distribution function is decomposed as $f=M_{f}+g$, where $M_{f}$ is the Maxwellian with the same first three velocity moments as $f$; i.e., $\rho_{f}=\langle M_{f} \mathbf{e} \rangle=\langle f \mathbf{e} \rangle$, with $\mathbf{e}=(1,v,\frac{1}{2}v^{2})^{\intercal}$.
The first three velocity moments of the perturbation $g$ satisfy $\langle g\mathbf{e}\rangle=0$.

The micro-macro method solves for a coupled system of equations for the moments $\rho_{f}$ and the perturbation $g$.
The code provided in this repository uses the discontinous Galerkin method for phase-space discretization, combined with implicit-explicit time integration.  

## Basic Code Description

All python source code is located under the _Source_ directory.

Drivers for the __Relaxation__, __Riemann__, and __Collisional Landau Damping__ problems are named _RelaxationMM.py_, _RiemannProblemMM.py_, and _CollisionalLandauDampingMM.py_, respectively.
Corresponding drivers using the direct method (solving for the kinetic distribution $f$) are named _Relaxation.py_, _RiemannProblem.py_, and _CollisonalLandauDamping.py_.  

## Running the Code