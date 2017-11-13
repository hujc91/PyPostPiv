from . import piv
from . import basics
from . import turbulence

import h5py
import warnings
import numpy as np

def load(file_path):
    h5file = h5py.File(file_path, 'r')
    grp = h5file['field']
    field_class = Field2D(grp.attrs['dt'], grp.attrs['x'], grp.attrs['y'], grp[0], grp[1])
    h5file.close()
    return field_class

def convert_vc7(vc7_folder_path, dt):
    """Converts a 2 dimensional 2 component VC7 file into the HDF5 format.

    Author(s)
    ---------
    Jia Cheng Hu
    """
    # Import all the nessessary libraries
    import ReadIM
    import glob

    # Get all file path
    all_vc7_path = glob.glob(vc7_folder_path+'*.vc7')

    # Get information of the first frames for Initialization
    first_vbuff, first_vattr = ReadIM.get_Buffer_andAttributeList(all_vc7_path[0])
    first_vattr_dict = ReadIM.att2dict(first_vattr)

    # Initialize storage dictionary for each camera
    data_all_cam = []
    for n_cam in range(first_vbuff.nf):

        u = np.zeros((first_vbuff.ny, first_vbuff.nx, len(all_vc7_path)))
        v = np.zeros((first_vbuff.ny, first_vbuff.nx, len(all_vc7_path)))

        dx =  float(first_vattr_dict['FrameScaleX'+str(n_cam)].splitlines()[0])*first_vbuff.vectorGrid/1000
        dy = -float(first_vattr_dict['FrameScaleY'+str(n_cam)].splitlines()[0])*first_vbuff.vectorGrid/1000

        x0 = float(first_vattr_dict['FrameScaleX'+str(n_cam)].splitlines()[1])/1000
        y0 = float(first_vattr_dict['FrameScaleY'+str(n_cam)].splitlines()[1])/1000

        x = x0 + np.arange(first_vbuff.nx)*dx + dx/2
        y = y0 - np.arange(first_vbuff.ny)*dy - dy/2

        xx, yy = np.meshgrid(x, y, indexing='ij')

        data_all_cam.append(piv.Field2D(dt, xx, yy, u, v))

    #Load velocity vector fields
    for i, vc7_path in enumerate(all_vc7_path):
        vbuff, vattr = ReadIM.get_Buffer_andAttributeList(vc7_path)
        v_array = ReadIM.buffer_as_array(vbuff)[0]

        for n_cam, data in enumerate(data_all_cam):
            # PIV Mask
            mask = np.ones((first_vbuff.ny, first_vbuff.nx))
            mask[v_array[n_cam*10] == 0] = np.nan

            # Vector scaling
            scaleI = float(ReadIM.att2dict(vattr)['FrameScaleI'+str(n_cam)].splitlines()[0])

            # Load velocity
            data[0,:,:,i] = v_array[1+n_cam*10]*scaleI*mask
            data[1,:,:,i] = v_array[2+n_cam*10]*scaleI*mask

            data[0,:,:,i] =  data[0,:,:,i].T
            data[1,:,:,i] = -data[1,:,:,i].T

    return tuple(data_all_cam)

class Field2D(np.ndarray):

    # Class Initialization -----------------------------------------------------
    def __new__(cls, *arg):
        obj = np.array(arg[3:]).view(cls)
        obj.dt = arg[0]
        obj.x = arg[1]
        obj.y = arg[2]
        obj.dL = np.abs(obj.x[0,0] - obj.x[1,0])

        if len(obj.shape) == 6:
            obj = obj[:,:,:,np.newaxis]
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.x = getattr(obj, 'x', None)
        self.y = getattr(obj, 'y', None)
        self.dt = getattr(obj, 'dt', None)
        self.dL = getattr(obj, 'dL', None)

    # Class Methods ------------------------------------------------------------
    def u(self, axis, time=None):
        if time == None:
            return self[axis:axis+1]
        else:
            return self[axis:axis+1, :, :, time:time+1]

    def get_value(self, axis, time):
        return self.x, self.y, np.array(self[axis, :, :, time])

    def len(self, dimension):
        if dimension == 'x': return self.shape[1]
        if dimension == 'y': return self.shape[2]
        if dimension == 't': return self.shape[3]

    def ftype(self):
        field = ('Scalar', '2D Vector', '3D Vector')
        return field[self.shape[0]-1]

    def save(self, file_path):
        f = h5py.File(file_path, 'w')
        f.create_dataset('field', data=self)
        f['field'].attrs['x'] = self.x
        f['field'].attrs['y'] = self.y
        f['field'].attrs['dt'] = self.dt
        f.close()

    def redim(self, s):
        return self[:, s:-s, s:-s]

    # Field Basic Operation ----------------------------------------------------
    def mag(self):
        return basics.mag(self)

    def mean(self):
        return basics.mean(self)

    def rms(self):
        return basics.rms(self)

    def ddx(self, method=None):
        return basics.ddx(self, method)

    def ddy(self, method=None):
        return basics.ddy(self, method)
