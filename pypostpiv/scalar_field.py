"""Defines scalar fields"""

import numpy as np

class scalar_field(np.ndarray):

    def __init__(self, t, x, y, scalar_field):
        self.t = t
        self.x = x
        self.y = y
        self.s = scalar_field

        self.dx = x[1]-x[0]
        self.dy = y[1]-y[0]
        self.dt = t[1]-t[0]
