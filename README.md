# ODE-Solver

ODE-solver using the method of multiple shooting

---


Table of contents
=================

<!--ts-->
   * [Prerequisites](#prerequisites)
   * [Short explanation](#short-explanation)
   * [Example](#example)
   * [Ideas for improvement](#ideas-for-improvement)
<!--te-->

Prerequisites
=====

The code has been tested with Python 2.7 and Python 3.
Libraries used:
* numpy
* matplotlib (if you want to run the examples provided)

Short explanation
=====
**Coming soon.** In the mean time [this](https://en.wikipedia.org/wiki/Direct_multiple_shooting_method) will be sufficient.


Example
=====
Consider the simple problem ![](./doc/bvp_poly.svg) which has the solution 
![](./doc/sol_poly.svg)

By simple substitution we can transform this problem to the form 


![](./doc/transformed_poly.svg)


Now we can turn this into code.

```python
import numpy as np  								# used to define vectors
import util											# used to generate initial values at each interval
import multish as msh  								# implementation of the multiple shooting algorithm
import rukutta as rk                                # implementation of explicit Runge-Kutta methods
import newtonssc									# implements different newton methods


def f(t, y):										# our  
    return np.array([y[1], t ** 2 + 1])


def r(a, b):
    return np.array([a[0] - 1, b[0] - 3])



if __name__ == "__main__":
    m = 10
    maxiter = 4
    t = np.linspace(0., 1., m + 1)

    bv_left = np.array([1, 0])
    bv_right = np.array([3, 0])
    interpolated_bv = util.getInterpolatedVectors(bv_right, bv_left, m + 1)

    Ba = np.matrix([[1, 0], [0, 0]])
    Bb = np.matrix([[0, 0], [1, 0]])

    integrator = rk.DormandPrince(h=.01)

    newtonSolver = newtonssc.SSCNewton(Ba=Ba, Bb=Bb, t=t, r=r,
                                       integrator=integrator, maxIter=0)
    t0 = time.time()
    shooter = msh.MultipleShootingIntegrator(t=t, m=m, dim=2,
                                             boundary_values=interpolated_bv,
                                             integrator=integrator,
                                             newtonSolver=newtonSolver)
    shooter.shoot(f=f, maxiter=maxiter, silent=False)


```
Ideas for improvement
=====
* Parallelization of the calculation of the jacobian matrix <math> ∇F(s(k)) </math>
* Rewriting the code in Cython
* Using the integrators provided by scipy
* Improving the Newton-methods

