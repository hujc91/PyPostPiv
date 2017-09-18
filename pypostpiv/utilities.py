"""A set of functions for loading, saving and general maintenance of data.
"""

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

def convert_2d_2c_vc7(vc7_folder_path, hdf5_file_path):
    """Converts a 2 dimensional 2 component VC7 file into the HDF5 format.

    Parameters
    ----------

    Author(s)
    ---------
    Jia Cheng Hu
    """
    # Import all the nessessary libraries
    import ReadIM
    import glob

    # Initialization
    all_vc7_path = glob.glob(vc7_folder_path+'*.vc7')

    # Initialize the data storage dictionary based on information of the first frames
    first_vbuff, first_vattr = ReadIM.get_Buffer_andAttributeList(all_vc7_path[0])
    first_vattr_dict = ReadIM.att2dict(first_vattr)

    # Initialize storage dictionary for each camera
    data = {}
    for n_cam in range(first_vbuff.nf):
        data['cam_'+str(n_cam)] = {}

    # Initialize storgage dictionary content
    for key, item in data.items():
        n_cam = key.split('_')[1]
        item['velocity'] = {}
        item['velocity']['u'] = np.zeros((first_vbuff.ny, first_vbuff.nx, len(all_vc7_path)))
        item['velocity']['v'] = np.zeros((first_vbuff.ny, first_vbuff.nx, len(all_vc7_path)))

        item['grid_size'] = {}
        item['grid_size']['dx'] = float(first_vattr_dict['FrameScaleX'+n_cam].splitlines()[0])\
                           *first_vbuff.vectorGrid/1000
        item['grid_size']['dy'] = -float(first_vattr_dict['FrameScaleY'+n_cam].splitlines()[0])\
                            *first_vbuff.vectorGrid/1000
        x0 = float(first_vattr_dict['FrameScaleX'+n_cam].splitlines()[1])/1000
        y0 = float(first_vattr_dict['FrameScaleY'+n_cam].splitlines()[1])/1000
        x = x0 + np.arange(first_vbuff.nx)*item['grid_size']['dx'] + item['grid_size']['dx']/2
        y = np.flipud(y0 - np.arange(first_vbuff.ny)*item['grid_size']['dy']) - item['grid_size']['dy']/2
        xx, yy = np.meshgrid(x,y)
        item['velocity']['x'] = xx
        item['velocity']['y'] = yy

    # Load velocity vector fields
    for i, vc7_path in enumerate(all_vc7_path):
        vbuff, vattr = ReadIM.get_Buffer_andAttributeList(vc7_path)
        v_array = ReadIM.buffer_as_array(vbuff)[0]

        for key, item in data.items():
            n_cam = key.split('_')[1]

            # PIV Mask
            mask = np.ones((first_vbuff.ny, first_vbuff.nx))
            mask[v_array[0+int(n_cam)*10] == 0] = np.nan

            # Vector scaling
            scaleI = float(ReadIM.att2dict(vattr)['FrameScaleI'+n_cam].splitlines()[0])

            # Load velocity
            item['velocity']['u'][:, :, i] = v_array[1+int(n_cam)*10]*scaleI*mask
            item['velocity']['v'][:, :, i] = v_array[2+int(n_cam)*10]*scaleI*mask

    # Invert camera y axis to be align with mesh grid orientation
    for key, item in data.items():
        item['velocity']['u'] = np.flipud(item['velocity']['u'])
        item['velocity']['v'] = -np.flipud(item['velocity']['v'])

    # Save into HDF5
    f_handle = h5py.File(hdf5_file_path, 'w')
    save_to_hdf5(f_handle, data)
    f_handle.close()

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
