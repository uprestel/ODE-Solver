# ODE-Solver

ODE-solver using the method of multiple shooting

---


Table of contents
=================

<!--ts-->
   * [Prerequisites](#prerequisites)
   * [Short explanation](#short-explanation)
   * [Examples](#examples)
      * [Simple ODE](#stdin)
      * [Lotka-Volterra](#Lotka-Volterra)
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
Coming soon. In the mean time [this][https://en.wikipedia.org/wiki/Direct_multiple_shooting_method] will be sufficient.


Examples
=====
Consider the simple problem ![alt text](./doc/bvp_poly.svg)



```python

```
Ideas for improvement
=====
* Parallelization of the calculation of the jacobian matrix <math> ∇F(s(k)) </math>
* Rewriting the code in Cython
* Using the integrators provided by scipy
* Improving the Newton-methods

