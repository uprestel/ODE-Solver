#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple implementation of the multiple shooting method to solve a two-point boundary-value problem
The problem dimension

To achieve this, we introduce m+1 artificial initial values s0, ..., sm

          t0=a    t1                        tm=b
            |     |                           |
            |     |      ...                  |
            |_____|___________________________|
            s_0  s_1      ...                s_m

and compute the solutions x(t_(n+1); t_n, s_n) from the initial value s_n at t_n.
Denote s(k) the vector of the initial values s(k)_0, ..., s(k)_m at the k-th iteration.
The goal is to minimize

           [               r(s(k)_0, s(k)_m)              ]
           |          x(t_1; t_0, s(k)_0) - s(k)_1        |
F(s(k)) =  |                   ...                        |
           | x(t_(m-1); t_(m-2), s(k)_(m-2)) - s(k)_(m-1) |
           [     x(t_m; t_(m-1), s(k)_(m-1)) - s(k)_m     ]


                [A    0    0    ...    0    B]          with G_j = ∂ x(t_(j+1); t_j, s(k)_j) /∂ s(k)_j
                |G_0 -I    0    ...    0    0|                 A = ∂ r(s(k)_0, s(k)_m) /∂ s(k)_0
∇F(s(k)) =      |0   G_1  -I    ...    0    0|                 B = ∂ r(s(k)_0, s(k)_m) /∂ s(k)_0
                |               ...          |
                [0 ...             G_(m-1) -I]

_________________________________________________________________________________________________
Sources used: http://www.mathsim.eu/~gkanscha/notes/ode.pdf                                      |
              https://wwwproxy.iwr.uni-heidelberg.de/~agbock/TEACHING/SKRIPTE/PEXTALKS/pex4.pdf  |
                                                                                                 |
                                                                                                 |
Author: Ulrich Prestel                                                                           |
_________________________________________________________________________________________________|
"""

import numpy as np


class MultipleShootingIntegrator(object):
    """
            Implements the multiple-shooting method to solve a BVP
        of the form
                        y'(t) = f(t, u(t))

        with boundary values at y(a), y(b) and t in [a,b]
    """

    def __init__(self, t, m, dim, boundary_values, integrator, newtonSolver):
        """
        Setup method.

        :param t: time-starting points of the intervals
        :param m: number of nodes
        :param dim: dimension of the problem
        :param boundary_values: initial values of m+1 nodes (start included!)
        :param integrator: integrator
        """

        self.m = m
        self.t = t
        self.initial_values = boundary_values
        self.integrator = integrator
        self.newtonSolver = newtonSolver

        self.d = dim

        self.times = []
        self.steps = []

    def shoot(self, f, tolerance=3e-16, maxiter=1000, silent=True):
        """
        Run the multiple shooting method.
        :param f: Right-hand-side function f(t,u)
        :param r: Boundary-Value function
        :param maxiter: Maximal number of iterations to run
        :param tolerance: Tolerance for how close we have to go to the zero of F(s(k))
        :param silent: Whether or not to print information about shooting
        """

        d = self.d

        exit = False

        s_k = np.zeros(self.m * d + d)
        x_k = np.zeros(self.m * d)
        F_k = np.array([1 for i in range(0, d * self.m)])

        for j in range(0, self.m + 1):
            for l in range(0, d):
                s_k[2 * j + l] = self.initial_values[j][l]

        for iteration in range(0, maxiter):

            for j in range(0, self.m):
                curr_t = self.t[j]
                t_lim = self.t[j + 1]

                initial_values = np.array([s_k[2 * j], s_k[2 * j + 1]])

                self.__solveIVP(f, initial_values, curr_t, t_lim)

                x_k[2 * j] = self.integrator.steps[-1][0]
                x_k[2 * j + 1] = self.integrator.steps[-1][1]

                if iteration == maxiter - 1 or np.linalg.norm(F_k, 2) < tolerance:

                    exit = True
                    for q in range(0, len(self.integrator.times)):
                        self.times.append(self.integrator.times[q])

                    for step in self.integrator.steps:
                        self.steps.append(step)

            if exit:
                if not silent:
                    print "finished multiple shooting after %s iterations" % iteration
                return None

            s_k = self.newtonSolver.update(f, s_k, x_k, d, self.m, self.__solveIVP)

    def __solveIVP(self, f, y0, t0, t):
        """
        Solves IVP u' = f(t,u) with initial values y0 at t0 for time t
        :param f: RHS function
        :param y0: initial values at t0
        :param t0: initial time t0
        :return: value of u at t
        """
        self.integrator.run(f, t0, y0, t_limit=t)
        return self.integrator.steps[-1]
