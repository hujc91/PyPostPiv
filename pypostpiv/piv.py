"""
Core utilities
"""

import os

# import warnings

import numpy as np
import h5py

# from . import piv
# from . import math
# from . import vorticity
# from . import turbulence

def load(file_path):
    """
    Returns a TensorField class from a 2D velocity field stored in an hdf5 file.

    Parameters
    ----------
    file_path : string

    Returns
    -------
    TensorField
    """
    field = None
    with h5py.File(file_path, 'r') as f:
        shape = f['field'].shape
        dx = (f['x'][1]-f['x'][0],
              f['y'][1]-f['y'][0])
        field = TensorField(shape[:-1], shape[-1:], dx=dx)
        field[...] = f['field'][...]
    return field

def convert_vc7(vc7_folder_path, dt):
    """
    Converts a 2 dimensional, 2 component VC7 file into the HDF5 format.

    Parameters
    ----------
    vc7_folder_path : string
        Path to a folder containing a collection of vc7 files.

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
    all_vc7_path = glob.glob(os.path.join(vc7_folder_path, '*.vc7'))

    # Get information of the first frames for Initialization
    first_vbuff, first_vattr = ReadIM.get_Buffer_andAttributeList(all_vc7_path[0])
    first_vattr_dict = ReadIM.att2dict(first_vattr)

    # Initialize storage dictionary for each camera
    data_all_cam = []
    for n_cam in range(first_vbuff.nf):

        u = np.zeros((first_vbuff.nx, first_vbuff.ny, len(all_vc7_path)))
        v = np.zeros((first_vbuff.nx, first_vbuff.ny, len(all_vc7_path)))

        dx = float(first_vattr_dict['FrameScaleX'+str(n_cam)]
                   .splitlines()[0])*first_vbuff.vectorGrid/1000
        dy = -float(first_vattr_dict['FrameScaleY'+str(n_cam)]
                    .splitlines()[0])*first_vbuff.vectorGrid/1000

        x0 = float(first_vattr_dict['FrameScaleX'+str(n_cam)]
                   .splitlines()[1])/1000
        y0 = float(first_vattr_dict['FrameScaleY'+str(n_cam)]
                   .splitlines()[1])/1000

        x = x0 + dx*(np.arange(first_vbuff.nx) + 1/2)
        y = y0 - dy*(np.arange(first_vbuff.ny) - 1/2)

        # xx, yy = np.meshgrid(x, y, indexing='ij')

        field_dx = (x[1]-x[0], y[1]-y[0])
        field = TensorField(u.shape, (2,), dx=field_dx)
        field[..., 0] = u
        field[..., 1] = v

        data_all_cam.append(field)

    # Load velocity vector fields
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
            data[0, ..., i] = (v_array[1+n_cam*10]*scaleI*mask).T
            data[1, ..., i] = -(v_array[2+n_cam*10]*scaleI*mask).T

    return tuple(data_all_cam)

class TensorField(np.ndarray):
    """
    A class representing a tensor field.

    Notes
    -----
    Two main attributes (_field_dx, _tensor_ndim) control how the array is interpreted into field
    and tensor portions. Operations that affect array shape must update these values in accordance
    with what the operation is meant to represent.

    Parameters
    ----------

    Attributes
    ----------
    _field_dx : tuple of floats and/or None
        Grid spacing for each of the field axes
    _tensor_ndim : int
        The dimension of the tensor
    _field_label : tuple of strings
        Probably would be useful to have?
    """
    def __new__(cls, field_shape, tensor_shape, dtype=float, buffer=None,
                offset=0, strides=None, order=None):
        obj = super(TensorField, cls).__new__(
            cls, field_shape+tensor_shape, dtype, buffer, offset, strides, order)

        obj._field_dx = tuple([None if obj.shape[ii] == 1 else 1 for ii in range(len(field_shape))])
        obj._tensor_ndim = len(tensor_shape)

        return obj

    def __array_finalize__(self, obj):
        # type(obj) differs based on the method of instance creation:
        # From explicit constructor -> None
        # From view casting -> any subclass of np.ndarray
        # From new-from template -> an instance of our own subclass of np.ndarray
        if obj is None:
            return

        # Set default values for TensorField attributes. We set default values unless the obj is an
        # instance of our own class, in which case we copy the objects attributes.
        self._field_dx = getattr(obj, 'field_dx', (1, )*obj.ndim)
        self._tensor_ndim = getattr(obj, 'tensor_ndim', 0)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Override logic for numpy's __array_ufunc__ method.

        There are two main classes of ufuncs: scalar ufuncs have
        'ufunc.signature is None' while generalized ufuncs (gufuncs) have a
        ufunc.signature string. These two cases are handled slightly differently.

        Parameters
        ----------
        ufunc : np.ufunc object
        method : str
            One of:
            '__call__' (indicates it was called directly)
            'accumulate'
            'reduceat'
            'outer'
            'at'
        inputs : tuple
            Input arguments to ufunc
        kwargs :
            Any additional arguments to ufunc

        Returns
        -------
        stuff
        """
        ## Conversion of inputs to ndarray types
        # First convert all inputs and outputs that are instances of our subclass
        # to views of ndarray and pass it off to the super class __array_ufunc__
        # method
        if kwargs.pop('out', None) is not None:
            # Haven't implemented how setting an output should work yet!
            return NotImplemented

        outputs = None
        converted_outputs = list()
        tensor_output_idxs = list()
        if outputs is None:
            converted_outputs = None
        else:
            for ii, output in enumerate(outputs):
                if isinstance(output, TensorField):
                    tensor_output_idxs.append(ii)
                    converted_outputs.append(output.view(np.ndarray))
                else:
                    converted_outputs.append(output)

        converted_inputs = list()
        tensor_input_idxs = list()
        tensor_inputs = list()
        for ii, _input in enumerate(inputs):
            if isinstance(_input, TensorField):
                tensor_inputs.append(_input)
                tensor_input_idxs.append(ii)
                converted_inputs.append(_input.view(np.ndarray))
            else:
                converted_inputs.append(_input)

        ## Determination of output field attributes/field compatibility
        # Loop through all TensorField inputs and broadcast their
        # field dimensions to an output TensorField dimension
        # Now broadcast the field dimensions
        dx = list()
        dxs = [tensor_input.field_dx for tensor_input in tensor_inputs]

        max_field_ndim = 0
        for _field_dx in dxs:
            if max_field_ndim < len(_field_dx):
                max_field_ndim = len(_field_dx)

        for ii in range(max_field_ndim):
            dx.append(None)
            for _field_dx in dxs:
                if len(_field_dx) < ii:
                    pass
                elif _field_dx[-1-ii] is None:
                    pass
                else:
                    if dx[ii] is None:
                        dx[ii] = _field_dx[-1-ii]
                    else:
                        if dx[ii] != _field_dx[-1-ii]:
                            return NotImplemented
        dx = list(reversed(dx))

        ## Perform the ufunc calculation
        _results = super(TensorField, self).__array_ufunc__(ufunc, method, *converted_inputs,
                                                            **kwargs)
        if _results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            _results = (_results, )

        ## Set output TensorField attributes
        results = list()
        if ufunc.signature is None:
            # For a scalar ufunc:
            # All inputs that are instances of our subclass must have the same
            # tensor order. Outputs will have the same tensor dimension
            tensor_ndim = tensor_inputs[0].tensor_ndim
            for tensor_input in tensor_inputs[1:]:
                if tensor_ndim != tensor_input.tensor_ndim:
                    return NotImplemented

            # The value of method (only occurs for scalar ufuncs) can influence
            # if any dimensions are created or lost.
            # Possible values are '__call__' 'reduce', 'reduceat', 'accumulate',
            # 'outer' and 'inner'
            # These methods can influence the TensorField attribute: dx,
            # tensor_ndim, blah blah blah
            if method == '__call__':
                # This doesn't affect shape so don't need to do anything
                pass
            elif method == 'reduce':
                # In this case, dimensions can be lost, changing the shape
                # of the broadcaster dimensions
                if kwargs['keepdims']:
                    pass
                    # In this case, the reduced dimension is left as a singular
                    # dimension. Maybe you want to make the dx for given axis
                    # a null value?
                else:
                    # We either have to get rid of one of the dx attribute entries,
                    # or one of the 'core tensor' entries
                    if kwargs['axis'] < len(dx):
                        dx.pop(kwargs['axis'])
                    else:
                        tensor_ndim = tensor_ndim-1
            elif method == 'reduceat':
                # don't know how this works do not going to do it!
                return NotImplemented
            elif method == 'accumulate':
                # This doesn't affect shape so don't need to do anything
                pass
            elif method == 'outer':
                # don't know how this works do not going to do it!
                return NotImplemented
            elif method == 'inner':
                # numpy documentation is a bit weird on this one... I assume this
                # corresponds to ufunc.at, in which case we don't have to modify
                # anything since this type of call preserves the shape
                pass
            else:
                print("Hi mom!")
                return NotImplemented

            for result in _results:
                result = result.view(TensorField)
                result._tensor_ndim = tensor_ndim
                result._field_dx = tuple(dx)
                results.append(result)
        else:
            # For a gufunc:
            # any inputs or outputs that are instances of our subclass
            # must have the tensor dimension equal to the number of core dimensions
            # for the corresponding input/output in ufunc.signature
            # The output tensor dimension(s) are the same as the corresponding
            # output core dimension(s) in ufunc.signature
            in_sigs, out_sigs = np.lib.function_base._parse_gufunc_signature(ufunc.signature)

            tensor_in_sigs = [in_sigs[ii] for ii in tensor_input_idxs]
            for tensor_input, tensor_in_sig in zip(tensor_inputs, tensor_in_sigs):
                if len(tensor_in_sig) != tensor_input.tensor_ndim:
                    return NotImplemented

            tensor_out_sigs = out_sigs
            for result, tensor_out_sig in zip(_results, tensor_out_sigs):
                result = result.view(TensorField)
                result._tensor_ndim = len(tensor_out_sig)
                result._field_dx = dx
                results.append(result)

        return results[0] if len(results) == 1 else results

    def __repr__(self):
        return f"TensorField({self.field_shape}, {self.tensor_shape})"

    ## Override shape influencing special methods
    # We also have to modify ndarray methods that can change the shape of the data. In these cases
    # we have to modify the field and/or tensor attributes
    # These are:
    # __get_item__,  __set_item__
    def __getitem__(self, indices):
        # Determine which indices have been sliced, and which have been integer
        # indexed. An integer index(s) will either remove elements from
        # _field_dx, or reduce the dimensionality of the tensor (depends where the
        # index is)
        if isinstance(indices, tuple):
            # Here we probably have a basic index. If we don't, let np.ndarray's
            # __getitem__ deal with it
            res = super(TensorField, self).__getitem__(indices)
            if not isinstance(res, TensorField):
                return res

            # Expand the indices with any ellipsis so that we have one entry for each index
            # The final length of the expanded indices depends on the number of np.newaxis (None)
            # objects in indices
            expanded_indices = None
            if Ellipsis in indices:
                ii = indices.index(Ellipsis)
                expanded_indices = indices[:ii] + \
                                   (Ellipsis, )*(self.ndim-len(indices)+1) \
                                   + indices[ii+1:]
            else:
                expanded_indices = indices + (slice(None), )*(self.ndim-len(indices))
            assert len(expanded_indices) == self.ndim

            # Part of expanded_indices index the field axes, while the remaining portion index the
            # tensor axes. Determine the index, ii, at which to split expanded_indices
            ii = 0
            count_field_index = 0
            while count_field_index < self.ndim-self.tensor_ndim:
                if expanded_indices[ii] is not None:
                    count_field_index += 1
                ii += 1
            field_indices = expanded_indices[:ii]
            tensor_indices = expanded_indices[ii:]

            dx = list()
            ii = 0
            for index in field_indices:
                if index is None:
                    dx.append(None)
                elif isinstance(index, int):
                    ii = ii+1
                else:
                    dx.append(self.field_dx[ii])
                    ii = ii+1
            #assert ii == self.field_ndim

            # Set attributes according to the indices
            res._field_dx = tuple(dx)

            is_none = [index for index in tensor_indices if index is None]
            is_int = [index for index in tensor_indices if isinstance(index, int)]
            res._tensor_ndim = self.tensor_ndim + len(is_none) - len(is_int)

            return res
        else:
            # Here we probably have an advanced index. If we don't, let ndarray's __getitem__ throw
            # an error
            # It's difficult to tell what kind of shape you'll get with advanced indexing, so always
            # return as an ndarray.
            return super(TensorField, self).__getitem__(indices).view(np.ndarray)

    def __setitem__(self, indices, value):
        ## Explicitly perform the data setting using a view of TensorField as
        # an ndarray.
        # See: https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        return np.ndarray.__setitem__(self.view(np.ndarray), indices, value)

    # as well as:
    def choose(self, choices, out=None, mode='raise'):
        raise NotImplementedError('TensorField.choose is not yet implemented.')

    def compress(self, condition, axis=None, out=None):
        raise NotImplementedError('TensorField.compress is not yet implemented.')

    def diagonal(self, offset=0, axis1=0, axis2=1):
        raise NotImplementedError('TensorField.diagonal is not yet implemented.')

    def flatten(self, order='C'):
        raise NotImplementedError('TensorField.flatten is not yet implemented.')

    def ravel(self, order='C'):
        raise NotImplementedError('TensorField.ravel is not yet implemented.')

    def repeat(self, repeats, axis=None):
        raise NotImplementedError('TensorField.repeat is not yet implemented.')

    def reshape(self, new_shape):
        raise NotImplementedError('TensorField.reshape is not yet implemented.')

    def resize(self, new_shape):
        raise NotImplementedError('TensorField.resize is not yet implemented.')

    def squeeze(self, axis=None):
        raise NotImplementedError('TensorField.squeeze is not yet implemented.')

    def swapaxes(self, axis1, axis2):
        raise NotImplementedError('TensorField.swapaxes is not yet implemented.')

    def take(self, indices, axis=None, out=None, mode='raise'):
        raise NotImplementedError('TensorField.take is not yet implemented.')

    def transpose(self, *axes):
        raise NotImplementedError('TensorField.transpose is not yet implemented.')

    @property
    def T(self):
        """Returns the transpose of the data."""
        raise NotImplementedError('TensorField.T is not yet implemented.')

    ## New methods
    def tensor_reshape(self, shape):
        """Reshapes the tensor portion of the array."""

    def field_reshape(self, shape):
        """Reshapes the field portion of the array."""

    ## Properties
    @property
    def field_dx(self):
        """
        Returns the field grid spacing data.
        """
        return self._field_dx

    @property
    def field_shape(self):
        """
        Returns the shape of the field.
        """
        return self.shape[:-self.tensor_ndim]

    @property
    def tensor_shape(self):
        """
        Returns the shape of the tensor.
        """
        return self.shape[-self.tensor_ndim:]

    @property
    def field_ndim(self):
        """
        Returns the dimension of the field.
        """
        return self.ndim - self.tensor_ndim

    @property
    def tensor_ndim(self):
        """
        Returns the dimension (order) of the tensor.
        """
        return self._tensor_ndim

    # The following are convenience properties for the common use case of 3 dimensional velocity
    # fields
    @property
    def u(self):
        """
        Returns the u component of a vector field.
        """
        if self.tensor_ndim == 1:
            return self[..., 0]
        else:
            raise NotImplementedError(
                f"'u' doesn't exist for order {self.tensor_ndim} tensor.")

    @property
    def v(self):
        """
        Returns the v component of a vector field.
        """
        if self.tensor_ndim != 1:
            raise NotImplementedError(
                f"'v' doesn't exist for order {self.tensor_ndim} tensor.")
        elif self.tensor_shape[0] < 1:
            raise NotImplementedError(
                f"'v' doesn't exist for {self.tensor_shape[0]}D vector.")
        else:
            return self[..., 1]

    @property
    def w(self):
        """
        Returns the w component of a vector field.
        """
        if self.tensor_ndim != 1:
            raise NotImplementedError(
                f"'w' doesn't exist for order {self.tensor_ndim} tensor.")
        elif self.tensor_shape[0] < 2:
            raise NotImplementedError(
                f"'w' doesn't exist for {self.tensor_shape[0]}D vector.")
        else:
            return self[..., 2]

    @property
    def dx(self):
        """
        Returns the 'x' field grid spacing for a <3D field.
        """
        if self.field_ndim > 3 or self.field_ndim < 1:
            raise NotImplementedError(
                f"'dx' is undefined for {self.field_ndim} dimensional field")
        else:
            return self.field_dx[0]

    @property
    def dy(self):
        """
        Returns the 'y' field grid spacing for a <3D field.
        """
        if self.field_ndim > 3 or self.field_ndim < 2:
            raise NotImplementedError(
                f"'dy' is undefined for {self.field_ndim} dimensional field")
        else:
            return self.field_dx[1]

    @property
    def dz(self):
        """
        Returns the 'z' field grid spacing for a <3D field.
        """
        if self.field_ndim != 3:
            raise NotImplementedError(
                f"'dz' is undefined for {self.field_ndim} dimensional field")
        else:
            return self.field_dx[2]
