"""
Implementations of Newton methods using different strategies to achieve globalization and efficiency

_________________________________________________________________________________________________
Sources used: http://www.mathsim.eu/~gkanscha/notes/ode.pdf                                      |
              https://wwwproxy.iwr.uni-heidelberg.de/~agbock/TEACHING/SKRIPTE/PEXTALKS/pex4.pdf  |
                                                                                                 |
                                                                                                 |
Author: Ulrich Prestel                                                                           |
_________________________________________________________________________________________________|
"""

import newtonmethod
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


class StdNewton(newtonmethod.NewtonMethod):
    """
    Standard newton implementation
    """

    def __init__(self, Ba, Bb, t, integrator, r, delta=0.0002):
        super(StdNewton, self).__init__(Ba=Ba, Bb=Bb, t=t, integrator=integrator, r=r, delta=delta)

    def update(self, f, s_k, x_k, d, m, solveIVP):
        jac = self.generateJacobian(f=f, s_k=s_k, x_k=x_k, d=d, m=m, solveIVP=solveIVP)
        jac_inv = np.linalg.inv(jac)
        F_k = self.generateF(s_k=s_k, x_k=x_k, d=d, m=m)
        return s_k - np.dot(jac_inv, F_k)


class EffNewton(newtonmethod.NewtonMethod):
    """
    Efficient newton implementation ( see remark 4.3.2 in the lecture notes )
    """

    def __init__(self, Ba, Bb, t, integrator, r, delta=0.0002, eta=0.5):
        super(EffNewton, self).__init__(Ba=Ba, Bb=Bb, t=t, integrator=integrator, r=r, delta=delta)
        self.eta = eta
        self.oldJacInv = None

    def update(self, f, s_k, x_k, d, m, solveIVP):
        if self.oldJacInv == None:
            return self.__step(f, s_k, x_k, d, m, solveIVP)
        else:
            F_k = self.generateF(s_k=s_k, x_k=x_k, d=d, m=m)
            s_hat = s_k - np.dot(self.oldJacInv, F_k)
            F_hat = self.generateF(s_hat, x_k, d, m)
            if np.linalg.norm(F_hat, 2) <= self.eta * np.linalg.norm(F_k, 2):
                return s_hat
            else:
                return self.__step(f, s_k, x_k, d, m, solveIVP)

    def __step(self, f, s_k, x_k, d, m, solveIVP):
        jac = self.generateJacobian(f=f, s_k=s_k, x_k=x_k, d=d, m=m, solveIVP=solveIVP)
        jac_inv = np.linalg.inv(jac)
        F_k = self.generateF(s_k=s_k, x_k=x_k, d=d, m=m)
        self.oldJacInv = jac_inv
        return s_k - np.dot(jac_inv, F_k)


class SSCNewton(newtonmethod.NewtonMethod):
    """
    Newton with step size control (see lecture notes, definition 4.2.3)
    """

    def __init__(self, Ba, Bb, t, integrator, r, maxIter=1074, delta=0.0002):
        super(SSCNewton, self).__init__(Ba=Ba, Bb=Bb, t=t, integrator=integrator, r=r, delta=delta)
        self.maxIter = maxIter

    def update(self, f, s_k, x_k, d, m, solveIVP):
        jac = self.generateJacobian(f=f, s_k=s_k, x_k=x_k, d=d, m=m, solveIVP=solveIVP)
        jac_inv = np.linalg.inv(jac)
        F_k = self.generateF(s_k=s_k, x_k=x_k, d=d, m=m)

        d_k = np.dot(jac_inv, F_k)
        alpha = self.__getexponent(F_k, s_k, x_k, d_k, d, m)
        return s_k - 2. ** -alpha * d_k

    def __getexponent(self, F_k, s_k, x_k, d_k, d, m):
        """
        Returns the step-size exponent for the newton iteration.
        :param F_k: d * m dimensional np.array holding all the values of F(s_(k))
        :param s_k: d * m dimensional np.array holding all the current guesses.
        :param x_k:  d * m dimensional np.array holding all the curretn solutions
        :param r: boundary value function
        :param d_k: new solution
        :return: maximum exponent
        """
        norm_Fk = np.linalg.norm(F_k, 2)
        for j in range(0, self.maxIter):
            F_k_hat = self.generateF(s_k - 2. ** -j * d_k, x_k, d=d, m=m)
            if np.linalg.norm(F_k_hat, 2) < norm_Fk:
                return j

        return self.maxIter
