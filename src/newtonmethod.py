import numpy as np


class NewtonMethod(object):
    """
    Framework for newton solvers
    """

    def __init__(self, Ba, Bb, t, integrator, r, delta=0.0002):
        """
        :param Ba: Boundary matrix for the first time point
        :param Bb: Boundary matrix for the second time point
        :param t: Starting-times of the intervals
        :param integrator: ODE integrator
        :param r: boundary value function
        :param delta: value to evaluate the finite difference g'(x) ~ [ g(x+delta) - g(x) } / delta
        """
        self.Ba = Ba
        self.Bb = Bb
        self.r = r
        self.integrator = integrator
        self.delta = delta
        self.t = t

    def update(self, f, s_k, x_k, d, m, solveIVP):
        """
        Updating method to generate the new value of s_k
        :param f: Right-hand-side function of the IVP u'(t) = f(t,u(t))
        :param s_k: d * m dimensional np.array holding all the current guesses.
        :param x_k:  d * m dimensional np.array holding all the curretn solutions
        :param d: dimension of the problem
        :param m: number of subintervals
        :param solveIVP: function to solve the IVP u'(t) = f(t,u(t)) for y0 at t0
        :return: updated s_k
        """
        raise NotImplementedError("You need to imlement a step-function!")

    def embedMatrix(self, A, B, i, j):
        """
        Helper function to embed a matrix B into matrix A
        :param A: Matrix to receive embedding
        :param B: Matrix to be embedded
        """
        A[i:i + B.shape[0], j: j + B.shape[1]] = B

    def generateJacobian(self, f, s_k, x_k, d, m, solveIVP):
        """
        generates the jacobian with special (sparse!) structure for the multiple shooting method
        :param s_k: d * m dimensional np.array holding all the current guesses.
        :param x_k:  d * m dimensional np.array holding all the curretn solutions
        :param d: dimension of the problem
        :param m: number of subintervals
        :param delta: the delta for finite difference
        :return:
        """

        jacobian = (-1) * np.identity(d * (m + 1))
        self.embedMatrix(jacobian, self.Ba, 0, 0)
        self.embedMatrix(jacobian, self.Bb, 0, d * m)

        for i in range(0, m):

            local_jac = np.zeros((d, d))
            current_s = np.array([s_k[2 * i], s_k[2 * i + 1]])
            current_x = np.array([x_k[2 * i], x_k[2 * i + 1]])
            tkp1 = self.t[i + 1]
            tk = self.t[i]

            for j in range(0, d):

                e_j = np.zeros(d)
                e_j[j] = 1
                y0 = current_s + self.delta * e_j

                delta_solution = solveIVP(f, y0, tk, tkp1)
                for k in range(0, d):
                    local_jac[k][j] = (delta_solution[k] - current_x[k]) / self.delta

            self.embedMatrix(jacobian, local_jac, i * d + d, i * d)

        return jacobian

    def generateF(self, s_k, x_k, d, m):
        """
        generates F(s_(k))
        :param s_k: d * m dimensional np.array holding all the current guesses.
        :param x_k: d * m dimensional np.array holding all the curretn solutions
        :return: d * m dimensional np.array holding all the values of F(s_(k))
        """
        F_k = np.zeros(m * d + d)
        s0 = np.array([s_k[0], s_k[1]])
        sm = np.array([s_k[-2], s_k[-1]])
        r_k = self.r(s0, sm)
        F_k[0] = r_k[0]
        F_k[1] = r_k[1]

        for i in range(0, m * d):
            F_k[i + d] = x_k[i] - s_k[i + d]

        return F_k
