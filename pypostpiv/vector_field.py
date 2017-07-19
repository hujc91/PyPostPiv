
""""Contains definitions of piv vector fields. """

import numpy as np

class VectorField2D(np.ndarray):

    def __init__(self, t, x, y, u, v):
        self.time = t
        self.x = x
        self.y = y

        self.u = u
        self.v = v
        self.dt = t[1] - t[0]
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
