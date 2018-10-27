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


class EmbeddedRungeKutta(ODEMethod):
    """
    RungeKutta method for given butcher tableau.
    """

    def __init__(self, A, b, c, bhat, order, epsilon=1e-7, h0=0.01, hmin=1e-10, hmax=1., beta=1.11, maxIter=50):
        """
        :param A: A- matrix from the butcher- tableau
        :param b: b- vector from the butcher- tableau
        :param c: c- vector from the butcher- tableau
        :param bhat: second b- vector from the butcher- tableau
        :param order: The highest order of the method
        :param epsilon: desired accuracy
        :param h0: initial step size
        :param hmin: minimal step size
        :param hmax: maximum step size
        :param beta: sensitivity constant
        :param maxIter: maximal number of iterations we update
        """
        self.A = A
        self.b = b
        self.bhat = bhat
        self.hmin = hmin
        self.hmax = hmax
        self.order = order
        self.c = c
        self.epsilon = epsilon
        self.beta = beta
        self.maxIter = maxIter
        self.initial_h = h0
        super(EmbeddedRungeKutta, self).__init__(step_len=h0)

    def step(self, f, i, t, y):
        """
        Compute one step of this method using adaptive step-size.

        Arguments:
            f    (callable)  Right hand side of ODE in standard form.
            i    (int)       Iteration index.
            t    (float)     Time at current step.
            y    (np.array)  Value at current step.

        Returns:
            t (float)    Next time.
            y (np.array) Next value.
        """

        for iteration in range(0, self.maxIter):
            k = np.array([f(t, y)]).reshape((len(y), 1))

            for s in range(1, self.A.shape[0]):
                ts = t + self.c[s] * self.h

                ys = y + self.h * np.matmul(k, self.A[s, :s])
                ks = f(ts, ys).reshape((len(y), 1))
                k = np.concatenate((k, ks), axis=1)

            ynext_hat = y + self.h * np.inner(k, self.bhat)
            ynext = y + self.h * np.inner(k, self.b)
            tnext = t + self.h

            epsilon0 = np.linalg.norm(ynext - ynext_hat, 2)
            if epsilon0 == 0.0:
                return tnext, ynext

            hopt = self.beta * self.h * (self.epsilon / epsilon0) ** (1. / (1 + self.order))

            if hopt < self.h:
                self.h = min(max(hopt, self.hmin), self.hmax)
                if iteration == self.maxIter - 1:
                    return tnext, ynext
                else:
                    continue
            else:
                self.h = min(max(hopt, self.hmin), self.hmax)

                if t + self.h > self.t_limit:
                    self.h = self.t_limit - t

                return tnext, ynext


class DormandPrince(EmbeddedRungeKutta):
    """
    Implementation of the embedded Dormand-Prince method (of order 4,5)
    """

    def __init__(self, h0=0.01, hmin=1e-10, hmax=1., epsilon=1e-7, beta=1.11, maxIter=10):
        """
        :param epsilon: desired accuracy
        :param h0: initial step size
        :param hmin: minimal step size
        :param hmax: maximum step size
        :param beta: sensitivity constant
        :param maxIter: maximal number of iterations we update
        """
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
        bhat = np.array([5179. / 57600, 0, 7571. / 16695, 393. / 640, -92097. / 339200, 187. / 2100, 1. / 40])

        c = np.array([0., 1. / 5, 3. / 10, 4. / 5, 8. / 9, 1., 1.])
        super(DormandPrince, self).__init__(A=A, b=b, bhat=bhat, c=c, order=5, h0=h0, hmin=hmin, hmax=hmax,
                                            epsilon=epsilon, beta=beta, maxIter=maxIter)
