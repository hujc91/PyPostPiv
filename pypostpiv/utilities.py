"""A set of functions for loading, saving and general maintenance of data.
"""

from . import piv
import h5py
import numpy as np

def save_to_hdf5(h5_handle, py_dict):
    """Saves all data from a dictionary to an HDF5 file.

    This function will save the data in the same structure and layout as the
    dictionary.

    Parameters
    ----------
    h5_handle : `h5py._hl.files.File`
        An HDF5 file handle that you want to load to as an input.

    Author(s)
    ---------
    Jia Cheng Hu
    """

    for key, value in py_dict.items():
        if not isinstance(value, dict):
            if isinstance(value, np.ndarray):
                h5_handle.create_dataset(key, data=value, compression="gzip")
            else:
                h5_handle.create_dataset(key, data=value)
        else:
            h5_handle.create_group(key)
            save_to_hdf5(h5_handle[key], value)

def load_from_hdf5(h5_handle):
    """Loads all data from an HDF5 file to a dictionary.

    This function will load the data in the same structure and layout as the
    HDF5 groups.

    Parameters
    ----------
    h5_handle : `h5py._hl.files.File`
        An HDF5 file handle that you want to load to as an input.

    Returns
    -------

    Author(s)
    ---------
    Jia Cheng Hu
    """

    def _load_from_hdf5(h5_handle, dict_h5):
        for key, value in h5_handle.items():
            if isinstance(value, h5py.Dataset):
                dict_h5[key] = value.value
            else:
                dict_h5[key] = {}
                dict_h5[key] = _load_from_hdf5(value, dict_h5[key])
        return dict_h5

    return _load_from_hdf5(h5_handle, dict())

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

        xx, yy = np.meshgrid(x,y, indexing='ij')

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

    return tuple(data_all_cam)

def print_dict_struct(py_dict, level=0):
    """Prints the structure of the given dictionary.

    Parameters
    ----------

    Author(s)
    ---------
    Jia Cheng Hu
    """

    for key, value in py_dict.items():
        if isinstance(value, dict):
            print('----'*level + key)
            print_dict_struct(value, level+1)
        else:
            print('----'*level + key)
