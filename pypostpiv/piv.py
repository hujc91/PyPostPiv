from . import piv
from . import basics
from . import vortex
from . import turbulence

import h5py
import warnings
import numpy as np

def load(file_path):
    """
    Loads a Field2D class stored in an HDF5 file.

    Parameters
    ----------
    file_path : string

    Returns
    -------
    Field2D
    """
    h5file = h5py.File(file_path, 'r')
    field_class = Field2D(
        h5file['dt'][()], h5file['x'][:],
        h5file['y'][:], h5file['field'][0], h5file['field'][1])
    h5file.close()
    return field_class

def convert_vc7(vc7_folder_path, dt):
    """Converts a 2 dimensional, 2 component VC7 file into the HDF5 format.

    Parameters
    ----------
    vc7_folder_path : string

    Returns
    -------
    tuple

    Author(s)
    ---------
    Jia Cheng Hu
    """
    # Import all the nessessary libraries
    import ReadIM
    import glob

    # Get all file path
    all_vc7_path = glob.glob(vc7_folder_path+'/*.vc7')

    # Get information of the first frames for Initialization
    first_vbuff, first_vattr = ReadIM.get_Buffer_andAttributeList(all_vc7_path[0])
    first_vattr_dict = ReadIM.att2dict(first_vattr)

    # Initialize storage dictionary for each camera
    data_all_cam = []
    for n_cam in range(first_vbuff.nf):

        u = np.zeros((first_vbuff.nx, first_vbuff.ny, len(all_vc7_path)))
        v = np.zeros((first_vbuff.nx, first_vbuff.ny, len(all_vc7_path)))

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
            data[0,:,:,i] =  (v_array[1+n_cam*10]*scaleI*mask).T
            data[1,:,:,i] = -(v_array[2+n_cam*10]*scaleI*mask).T

    return tuple(data_all_cam)

def vector(*args):
    """
    Combines a set of scalar fields into a vector field.

    Parameters
    ----------
    *args : array_like
        a set of scalar fields

    Returns
    -------
    Field2D

    Author(s)
    ---------
    Jia Cheng Hu
    """
    if len(args) == 2:
        if args[0].ftype() is 'scalar' and args[1].ftype() is 'scalar':
            new_field = Field2D(
                args[0].dt, args[0].x, args[0].y,
                np.array(args[0][0]), np.array(args[1][0]))
            return new_field

class Field2D(np.ndarray):
    """
    This class represents a 2D vector field in time and space.

    All the processing functions (operating only on on field argument) have
    been added as methods to the class. For example:
    pypostpiv.basics.ddx(field_instance) can also be accessed as
    field_instance.ddx() for convenience.
    """

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
        if obj is None:
            return None
        self.x = getattr(obj, 'x', None)
        self.y = getattr(obj, 'y', None)
        self.dt = getattr(obj, 'dt', None)
        self.dL = getattr(obj, 'dL', None)

    def u(self, axis, time=None):
        """
        Gets a component of the velocity field, 0 for u, 1 for v.

        This returns the field with the specific component of the velocity
        at all times. If the time argument is specified, only that time
        is returned.
        """
        if time is None:
            return self[axis:axis+1]
        else:
            return self[axis:axis+1, :, :, time:time+1]

    def get_value(self, axis=None, time=None):
        """
        Description.

        Parameters
        ----------

        Returns
        -------

        """
        if axis is None and time is None:
            return  self.x, self.y, np.array(self[0, :, :, 0])
        return self.x, self.y, np.array(self[axis, :, :, time])

    def len(self, dimension):
        """
        Description.

        Parameters
        ----------

        Returns
        -------

        """
        if dimension == 'x':
            return self.shape[1]
        elif dimension == 'y':
            return self.shape[2]
        elif dimension == 't':
            return self.shape[3]

    def ftype(self):
        """
        Description.

        Parameters
        ----------

        Returns
        -------

        """
        if self.shape[0] == 1:
            return 'scalar'
        elif self.shape[0] == 2:
            return 'vector'
        else:
            assert()

    def save(self, file_path):
        """
        Description.

        Parameters
        ----------

        Returns
        -------

        """
        f = h5py.File(file_path, 'w')
        f.create_dataset('field', data=self)
        f.create_dataset('x', data=self.x)
        f.create_dataset('y', data=self.y)
        f.create_dataset('dt', data=self.dt)
        f.close()

    def redim(self, s):
        """
        Description.

        Parameters
        ----------

        Returns
        -------

        """
        return self[:, s:-s, s:-s]

    # Field Basic Operations ---------------------------------------------------
    def fsum(self,axis):
        return basics.fsum(self,axis=axis)

    def mag(self):
        return basics.mag(self)

    def fmean(self):
        return basics.fmean(self)

    def rms(self):
        return basics.rms(self)

    def ddx(self, method=None):
        return basics.ddx(self, method)

    def ddy(self, method=None):
        return basics.ddy(self, method)

    # Turbulence Operations ----------------------------------------------------
    def turbulent_kinetic_energy(self):
        return turbulence.turbulent_kinetic_energy(self)

    def reynolds_shear_stress(self):
        return turbulence.reynolds_shear_stress(self)

    # Vortex dynamics ----------------------------------------------------------
    def vorticity(self, method=None):
        return vortex.vorticity(self, method)

    def lambda2(self, method=None):
        return vortex.lambda2(self, method)
