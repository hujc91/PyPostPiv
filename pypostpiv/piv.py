from . import utilities
from . import turbulence

import numpy as np

class Field2D(np.ndarray):

    def __new__(cls, *arg):

        if len(arg) == 1:
            pass
        else:
            obj = np.array(arg[3:]).view(cls)
            obj.dt = arg[0]
            obj.x = arg[1]
            obj.y = arg[2]
            obj.dL = obj.x[0,0] - obj.x[0,1]
            if len(obj.shape) == 6:
                obj = obj[:,:,:,np.newaxis]
            return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.x = getattr(obj, 'x', None)
        self.y = getattr(obj, 'y', None)
        self.dt = getattr(obj, 'dt', None)
        self.dL = getattr(obj, 'dL', None)

    def u(self, axis, time=None):
        if time == None:
            return self[axis:axis+1]
        else:
            return self[axis:axis+1, :, :, time:time+1]

    def len(self, dimension):
        if dimension == 'x': return self.shape[1]
        if dimension == 'y': return self.shape[2]
        if dimension == 't': return self.shape[3]

    def ftype(self):
        field = ('Scalar', '2D Vector', '3D Vector')
        return field[self.shape[0]-1]
