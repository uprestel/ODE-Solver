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

                ( t * u_2(t)          )
        u'(t) = ( 4 * u_1(t) ^ (3/2)  )      for t ∈ [0,5]

        u_1(0) = 1,   u_1(5) = 0
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
    return np.array([t * y[1], 4. * y[0] ** 3. / 2.])


def r(s0, sm):
    return np.array([s0[0] - 1, sm[0]])


if __name__ == "__main__":
    # ------- This is an example to demonstrate globalization for Thomas-Fermi -------
    m = 20
    maxiter = 60                      # We need more iterations to reach a 'smooth' solution!
    t = np.linspace(0., 5., m + 1)

    bv_left = np.array([1, -10])      # New Initial value. The regular Newton-Solver will run into a singular jacobian!
    bv_right = np.array([0, 0])
    interpolated_bv = util.getInterpolatedVectors(bv_right, bv_left, m + 1)

    Ba = np.matrix([[1, 0], [0, 0]])
    Bb = np.matrix([[0, 0], [1, 0]])
    integrator = erk.DormandPrince(h0=.1, hmin=0.1, hmax=0.3, epsilon=1e-5)

    # If you try running it with these regular newton-solvers, an exception will be thrown!
    # newtonSolver = newtonssc.StdNewton(Ba=Ba, Bb=Bb, t=t, r=r,
    #                                   integrator=integrator)
    # newtonSolver = newtonssc.EffNewton(Ba=Ba, Bb=Bb, t=t, r=r,
    #                                  integrator=integrator)


    # We'll have to use this Newton-solver with (effective) step-size control!
    newtonSolver = newtonssc.SSCNewton(Ba=Ba, Bb=Bb, t=t, r=r,
                                      integrator=integrator, maxIter=3)

    print "starting multiple shooting"
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

    integrator.run(f, 0, np.array([1., -1.68538396]), 5)

    plt.subplot(grid[0, 1])
    plt.title(r"Recomputed solution with\\correct initial values and IVP solver")
    plt.plot(integrator.times, [i[0] for i in integrator.steps], c="m")
    plt.ylabel(r'$u_1(t)$')

    plt.subplot(grid[1, 1])
    plt.plot(integrator.times, [i[1] for i in integrator.steps], c="m")
    plt.ylabel(r'$u_2(t)$')
    plt.xlabel(r'\textbf{time} (s)')

    plt.show()
