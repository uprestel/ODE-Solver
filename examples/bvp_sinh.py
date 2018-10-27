#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
_________________________________________________________________________________________________
Sources used: http://www.mathsim.eu/~gkanscha/notes/ode.pdf                                      |
              https://wwwproxy.iwr.uni-heidelberg.de/~agbock/TEACHING/SKRIPTE/PEXTALKS/pex4.pdf  |
                                                                                                 |
                                                                                                 |
Author: Ulrich Prestel                                                                           |
_________________________________________________________________________________________________|

This example solves the Thomas- Fermi BVP

                (    u_2(t)          )
        u'(t) = ( λsinh(λu_1(t))     )       for t ∈ [0,1]

        u_1(0) = 1,   u_1(1) = 1

        For single shooting this problem is only solvable for λ < 2.
        We choose λ = 5 and plot the solution.
        This BVP is from the second source, page 19.

        The solution values / the plots in general should match up.
"""

import numpy as np
import util
import multish as msh
import matplotlib.pyplot as plt
import embrukutta as erk
import rukutta as rk
import time
import newtonssc
from math import sinh

def f(t, y):
    return np.array([y[1], 5 * sinh(5 * y[0])])


def r(s0, sm):
    return np.array([s0[0], sm[0] - 1])


if __name__ == "__main__":
    m = 20
    t = np.linspace(0., 1., m + 1)
    maxiter = 7
    l = [i for i in range(0, maxiter)]

    bv_left = np.array([0, 0])
    bv_right = np.array([1, 0])
    interpolated_bv = util.getInterpolatedVectors(bv_right, bv_left, m + 1)

    Ba = np.matrix([[1, 0], [0, 0]])
    Bb = np.matrix([[0, 0], [1, 0]])

    integrator = rk.DormandPrince(h=.05)
    # newtonSolver = newtonssc.StdNewton(Ba=Ba, Bb=Bb, t=t, r=r,
    #                                  integrator=integrator)

    # Efficient solver
    # newtonSolver = newtonssc.EffNewton(Ba=Ba, Bb=Bb, t=t, r=r,
    #                                  integrator=integrator)

    # Newton solver with step-size control
    newtonSolver = newtonssc.SSCNewton(Ba=Ba, Bb=Bb, t=t, r=r,
                                      integrator=integrator, maxIter=0)


    print("starting multiple shooting")
    t0 = time.time()
    shooter = msh.MultipleShootingIntegrator(t=t, m=m, dim=2,
                                             boundary_values=interpolated_bv,
                                             integrator=integrator,
                                             newtonSolver=newtonSolver)
    shooter.shoot(f=f, maxiter=maxiter, silent=False)
    t1 = time.time()
    print("multiple shooting finished in %s seconds!" % (t1 - t0))

    # ----------------------- plotting ---------------------------------------------
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(13,6))

    grid = plt.GridSpec(2, 2, wspace=0.2, hspace=0.2)

    plt.subplot(grid[0, 0])
    plt.plot(shooter.times, [i[0] for i in shooter.steps])

    plt.title(r"Solution $u(t)$ of Thomas-Fermi BVP\\with multiple shooting")
    plt.ylabel(r'$u_1(t)$')

    plt.subplot(grid[1, 0])
    plt.plot(shooter.times, [i[1] for i in shooter.steps])
    plt.ylabel('$u_2(t)$')
    plt.xlabel(r'\textbf{time} (s)')

    integrator.run(f, 0, np.array([0, 4.57660620e-02]), 1)

    plt.subplot(grid[0, 1])
    plt.title(r"Recomputed solution with\\correct initial values and IVP solver")
    plt.plot(integrator.times, [i[0] for i in integrator.steps], c="m")
    plt.ylabel(r'$u_1(t)$')

    plt.subplot(grid[1, 1])
    plt.plot(integrator.times, [i[1] for i in integrator.steps], c="m")
    plt.ylabel(r'$u_2(t)$')
    plt.xlabel(r'\textbf{time} (s)')

    plt.show()