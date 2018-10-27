"""
_________________________________________________________________________________________________
Sources used: http://www.mathsim.eu/~gkanscha/notes/ode.pdf                                      |
              https://wwwproxy.iwr.uni-heidelberg.de/~agbock/TEACHING/SKRIPTE/PEXTALKS/pex4.pdf  |
                                                                                                 |
                                                                                                 |
Author: Ulrich Prestel                                                                           |
_________________________________________________________________________________________________|
"""

from odemethod import ODEMethod
import numpy as np


class RungeKutta(ODEMethod):
    """
    RungeKutta method for given butcher tableau.
    """

    def __init__(self, A, b, c, h=0.01):
        self.A = A
        self.b = b
        self.c = c
        super(RungeKutta, self).__init__(step_len=h)

    def step(self, f, i, t, y):
        """
        Compute one step of this method.

        Arguments:
            f    (callable)  Right hand side of ODE in standard form.
            i    (int)       Iteration index.
            t    (float)     Time at current step.
            y    (np.array)  Value at current step.

        Returns:
            t (float)    Next time.
            y (np.array) Next value.
        """

        k = np.array([f(t, y)]).reshape((len(y), 1))

        for s in range(1, self.A.shape[0]):
            ts = t + self.c[s] * self.h

            ys = y + self.h * np.matmul(k, self.A[s, :s])
            # print "eval at", ts, ys
            ks = f(ts, ys).reshape((len(y), 1))
            k = np.concatenate((k, ks), axis=1)

        return t + self.h, y + self.h * np.inner(k, self.b)


class Cash_Karp(RungeKutta):
    """
    Implementation of the Cash-Karp method (of order 4)
    """

    def __init__(self, h=0.01):
        A = np.array([
            [0, 0, 0, 0, 0, 0],
            [1. / 5, 0, 0, 0, 0, 0],
            [3. / 40, 9. / 40, 0, 0, 0, 0],
            [3. / 10, -9. / 10, 6. / 5, 0, 0, 0],
            [-11. / 54, 5. / 2, -70. / 27, 35. / 27, 0, 0],
            [1631. / 55296, 175. / 512, 575. / 13824, 44275. / 110592, 253. / 4096, 0]])
        b = np.array([2825. / 27648, 0, 18575. / 48384, 13525. / 55296, 277. / 14336, 1. / 4])
        c = np.array([0, 1. / 5, 3. / 10, 3. / 5, 1., 7. / 8])
        super(Cash_Karp, self).__init__(A, b, c, h=h)


class DormandPrince(RungeKutta):
    """
    Implementation of the Dormand-Prince method (of order 5)
    """

    def __init__(self, h=0.01):
        A = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [1. / 5, 0, 0, 0, 0, 0, 0],
            [3. / 40, 9. / 40, 0, 0, 0, 0, 0],
            [44. / 45, -56. / 15, 32. / 9, 0, 0, 0, 0],
            [19372. / 6561, -25360. / 2187, 64448. / 6561, -212. / 729, 0, 0, 0],
            [9017. / 3168, -355. / 33, 46732. / 5247, 49. / 176, -5103. / 18656, 0, 0],
            [35. / 384, 0, 500. / 1113, 125. / 192, -2187. / 6784, 11. / 84, 0]
        ])
        b = np.array([35. / 384, 0, 500. / 1113, 125. / 192, -2187. / 6784, 11. / 84, 0])
        c = np.array([0., 1. / 5, 3. / 10, 4. / 5, 8. / 9, 1., 1.])
        super(DormandPrince, self).__init__(A, b, c, h=h)
