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


def getInterpolatedVectors(v1, v2, m):
    """
    Generates m interpolated Vectors (np.array), where the start- and end-vector are included
    :param v1: first vector
    :param v2: second vector
    :param m: number of vectors
    :return: list of vectors (np.arrays)
    """
    vectors = []
    t_points = np.linspace(0., 1., m)
    for t in t_points:
        vectors.append(t * v1 + (1 - t) * v2)

    return vectors


def eval_function(f, domain):
    """
    Evaluares a function f (callable) over a domain and returns the values in an np.array
    :param f: function (callable)
    :param domain: domain (np.array)
    :return: np.array values
    """

    n = len(domain)
    values = np.zeros(n)
    for i in range(0, n):
        values[i] = f(domain[i])
    return values
