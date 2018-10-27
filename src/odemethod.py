"""
_________________________________________________________________________________________________
Sources used: http://www.mathsim.eu/~gkanscha/notes/ode.pdf                                      |
              https://wwwproxy.iwr.uni-heidelberg.de/~agbock/TEACHING/SKRIPTE/PEXTALKS/pex4.pdf  |
                                                                                                 |
                                                                                                 |
Author: Ulrich Prestel                                                                           |
_________________________________________________________________________________________________|
"""

import numpy as np

class ODEMethod(object):
    """
    Base class for IVP solver methods.
    The common wrapped functionality includes collecting of intermediate
    computation results.

    Attributes:
        times ([float]) 	Array of time steps. Default is [].
        steps ([np.array])  Array of computed steps. Default is [].
        h     (float)       Step size. Default is 1e-2.
        t0    (float)       Initial time. Default is 0.
        y0    (np.array)    Initial value. Default is 1.
    """

    def __init__(self, step_len=0.01):
        """
        Setup method.

        Arguments:
            step_len (float)     Step length. Default is 1e-2.
        """
        self.h = step_len

    def step(self, f, i, t, y):
        """
        Blueprint: Compute one step of this method.

        Arguments:
            f    (callable)  Right hand side of ODE in standard form.
            i    (int)       Iteration index.
            t    (float)     Time at current step.
            y    (np.array)  Value at current step.

        Returns:
            t (float)    Next time.
            y (np.array) Next value.
        """
        raise NotImplementedError("Please specify the step procedure in a child class (method implementation).")


    def run(self, f, t0=0.0, y0=1.0, t_limit=2.0):
        """
        Execute the method to solve an IVP of shape
            y' = f(t,y);    y(t0) = y0
        where len(y0) determines the problem dimension.

        Arguments:
            f 	       (callable)  Right hand side of ODE in standard form.
            t0         (float)     Initial time. Default is 0.
            y0         (np.array)  Initial value. Default is 1.
            t_limit    (float)     Limit on t to compute up to. Default is None.
        """
        self.t_limit = t_limit

        self.n = np.product(y0.shape) if type(y0) != float else 1
        self.t0 = t0
        if type(y0) == float:
            self.y0 = np.array([y0])
        else:
            self.y0 = y0.reshape((1, self.n))
        self.steps = np.zeros((0, self.n))
        self.steps = np.concatenate((self.steps, self.y0), axis=0)
        self.times = [t0]

        t = self.times[0]
        y = self.steps[0, :]
        i = 0
        while t <= t_limit - self.h:
            #print "here in odeloop", t, t_limit, self.h, t_limit-self.h
            (t, y) = self.step(f, i, t, y)
            i += 1
            self.times.append(t)
            self.steps = np.concatenate((self.steps, y.reshape((1, self.n))), axis=0)

        # guarantee to exactly hit the right interval end
        tmp_h = self.h
        self.h = t_limit - t
        (t, y) = self.step(f, i, t, y)
        self.h = tmp_h
        self.times.append(t)
        self.steps = np.concatenate((self.steps, y.reshape((1, self.n))), axis=0)
