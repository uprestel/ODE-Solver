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

This example solves a BVP

                ( u_2(t)  )
        u'(t) = ( t^2 + 1 )      for t ∈ [0,1]

        u_1(0) = 1,   u_1(1) = 3

        which has the exact solution u_1(t) = 1/12 t^4 + 1/2 t^2 + 17/12 t + 1
"""

import numpy as np
import util
import multish as msh
import matplotlib.pyplot as plt
import embrukutta as erk
import rukutta as rk
import time
import newtonssc


def f(t, y):
    return np.array([y[1], t ** 2 + 1])


def r(s0, sm):
    return np.array([s0[0] - 1, sm[0] - 3])


def exact_f(t):
    return 1. / 12 * t ** 4 + 1. / 2 * t ** 2 + 17. / 12 * t + 1


if __name__ == "__main__":
    m = 10
    maxiter = 4
    t = np.linspace(0., 1., m + 1)

    bv_left = np.array([1, 0])
    bv_right = np.array([3, 0])
    interpolated_bv = util.getInterpolatedVectors(bv_right, bv_left, m + 1)

    Ba = np.matrix([[1, 0], [0, 0]])
    Bb = np.matrix([[0, 0], [1, 0]])

    # integrator = rk.DormandPrince(h=.01)
    integrator = erk.DormandPrince(h0=.1, hmin=0.01, hmax=0.5, epsilon=1e-6)

    newtonSolver = newtonssc.SSCNewton(Ba=Ba, Bb=Bb, t=t, r=r,
                                       integrator=integrator, maxIter=0)
    t0 = time.time()
    shooter = msh.MultipleShootingIntegrator(t=t, m=m, dim=2,
                                             boundary_values=interpolated_bv,
                                             integrator=integrator,
                                             newtonSolver=newtonSolver)
    shooter.shoot(f=f, maxiter=maxiter, silent=False)
    t1 = time.time()
    print "multiple shooting finished in %s seconds!" % (t1 - t0)

    # ----------------------- plotting ---------------------------------------------
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(13, 6))

    grid = plt.GridSpec(3, 2, wspace=0.2, hspace=0.3)

    plt.subplot(grid[0, 0])
    plt.plot(shooter.times, [i[0] for i in shooter.steps])

    plt.title(r"Solution $u(t)$ of Polynomial BVP\\with multiple shooting")
    plt.ylabel(r'$u_1(t)$')

    plt.subplot(grid[1, 0])
    plt.plot(shooter.times, [i[1] for i in shooter.steps])
    plt.ylabel('$u_2(t)$')

    integrator.run(f, 0, np.array([1., 1.41666667]), 1)

    plt.subplot(grid[0, 1])
    plt.title(r"Recomputed solution with\\correct initial values and IVP solver")
    plt.plot(integrator.times, [i[0] for i in integrator.steps], c="m")
    plt.ylabel(r'$u_1(t)$')

    plt.subplot(grid[1, 1])
    plt.plot(integrator.times, [i[1] for i in integrator.steps], c="m")
    plt.ylabel(r'$u_2(t)$')

    exact = util.eval_function(exact_f, shooter.times)

    plt.subplot(grid[2, 0])
    plt.title(r"Exact solution")
    plt.plot(shooter.times, exact, c="r")
    plt.ylabel(r'$u(t)$')
    plt.xlabel(r'\textbf{time} (s)')

    error = []
    for i in range(0, len(exact)):
        error.append(abs(exact[i] - shooter.steps[i][0]))

    print "Max. error: %s \n Min. error %s" % (max(error), min(error))
    plt.subplot(grid[2, 1])
    plt.title(r"Error between multiple-sh. and exact solution")
    plt.plot(shooter.times, error, c="r")
    plt.xlabel(r'\textbf{time} (s)')

    plt.show()
