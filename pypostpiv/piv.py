"""
Core utilities
"""

import os

# import warnings
import itertools

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

def _expand_indices(indices, n):
    """
    Expands a set of implicit, numpy style, basic indices to n dimensions.

    Parameters
    ----------
    indices : tuple of slice, int, or None
    """
    is_not_none = [index for index in indices if index is not None]
    n_indexed_axes = len(is_not_none)
    if n_indexed_axes > n:
        raise IndexError("Too many indices")
    else:
        if ... in indices:
            ii = indices.index(...)
            return indices[:ii] + (slice(None),)*(n-n_indexed_axes+1) + indices[ii+1:]
        else:
            exp_indices = (slice(None),)*(n-n_indexed_axes)
            return indices + exp_indices

def _index_at_count(iterable, condition, n):
    """
    Returns the index in iterable, where the cumulative number of times func is True reaches
    count.
    """
    count = 0
    for ii, item in enumerate(iterable):
        if condition(item):
            count += 1
            if count == n:
                return ii
    return None

def _parse_label_indices():
    """
    Returns field and tensor indices given field labels and label style indices.
    """
    pass

def _broadcast(*args):
    """
    Returns an output tuple, where each entry in the list is determined from the elements in args
    with each arg in args right aligned.

    Parameters
    ----------
    broadcast_func(values) : function
        Returns a single value based on the values in the list. Otherwise return not implemented.
    """
    res = list()
    for values in itertools.zip_longest(*[reversed(arg) for arg in args]):
        if len(values) == 1:
            res.append(values[0])
        else:
            broadcasted_val = values[0]
            for val in values[1:]:
                if val != broadcasted_val and broadcasted_val is None:
                    broadcasted_val = val
                elif val is None or val == broadcasted_val:
                    pass
                    # Keep current value
                else:
                    # Have conflicting values so force it to None
                    broadcasted_val = None
                    break
            res.append(broadcasted_val)
    return list(reversed(res))

class TensorField(np.ndarray):
    """
    A class representing a tensor field.

    Notes
    -----
    Two main attributes (_field_dx, _tensor_ndim) control how the array is interpreted into field
    and tensor portions. Operations that affect array shape must should update these values
    depending on what the operation is meant to represent. For example, a norm over a tensor axis
    should reduce the tensor dimension by 1.

    For ufuncs, this is implemented in the __array_ufunc__ logic. Functions affecting shape that are
    not ufuncs must implement this manually.

    Parameters
    ----------

    Attributes
    ----------
    _tensor_ndim : int
        The dimension of the tensor
    _field_dx : tuple of floats and/or None
        Grid spacing for each of the field axes
    _field_labels : tuple of strings
        Probably would be useful to have?
    """
    def __new__(cls, field_shape, tensor_shape, dtype=float, buffer=None,
                offset=0, strides=None, order=None):
        obj = super(TensorField, cls).__new__(
            cls, field_shape+tensor_shape, dtype, buffer, offset, strides, order)

        obj._field_dx = (1.0,) * len(field_shape)
        obj._field_labels = (None,) * len(field_shape)
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
        self._field_labels = getattr(obj, 'field_labels', (None,)*obj.ndim)
        self._field_dx = getattr(obj, 'field_dx', (1,)*obj.ndim)
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
        inputs_field_dx = [tensor_input.field_dx for tensor_input in tensor_inputs]
        inputs_field_labels = [tensor_input.field_labels for tensor_input in tensor_inputs]

        field_labels = _broadcast(*inputs_field_labels)
        field_dx = _broadcast(*inputs_field_dx)

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
            # These methods can influence the TensorField attribute: field_dx,
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
                    # dimension. Maybe you want to make the field_dx for given axis
                    # a null value?
                else:
                    # We either have to get rid of one of the field_dx attribute entries,
                    # or one of the 'core tensor' entries
                    if kwargs['axis'] < len(field_dx):
                        field_dx.pop(kwargs['axis'])
                        field_labels.pop(kwargs['axis'])
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
                result._field_dx = tuple(field_dx)
                result._field_labels = tuple(field_labels)
                result._tensor_ndim = tensor_ndim
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
                result._field_dx = tuple(field_dx)
                result._field_labels = tuple(field_labels)
                result._tensor_ndim = len(tensor_out_sig)
                results.append(result)

        return results[0] if len(results) == 1 else results

    def __repr__(self):
        return f"TensorField({self.field_shape}, {self.tensor_shape})" + f"\n{np.ndarray.__repr__(self.view(np.ndarray))}"

    ## Override shape influencing special methods
    # We also have to modify ndarray methods that can change the shape of the data. In these cases
    # we have to modify the field and/or tensor attributes
    # These are:
    # __get_item__,  __set_item__
    def __getitem__(self, indices):
        has_strs = [index for index in indices if isinstance(index, str)]
        has_valid_slice = [index for index in indices if isinstance(index, (slice, int))
                                                         or index is None or index is ...]
        has_advanced_index = [index for index in indices if isinstance(index, (list, np.ndarray))]

        if len(has_valid_slice) == len(indices):
            ## Numpy Style Basic Indexing
            # There's an edge case with multiple new axes inserted between the field and tensor
            # dimensions, since it's not clear which ones go to the tensor and which ones to the
            # field. As a result, just assume they all go to the tensor
            # Calculate a split, ii, to split the indices into field and tensor portions
            indices = _expand_indices(indices, self.ndim)
            ii = _index_at_count(indices, lambda ind: ind is not None, self.field_ndim)

            field_indices = indices[:ii+1]
            tensor_indices = indices[ii+1:]
            return self._getitem_tf(field_indices, tensor_indices)
        elif has_advanced_index:
            ## Numpy Style Advanced Indexing
            # Here we probably have an advanced index.
            # I don't what kind of shape you'll get with advanced indexing, so always
            # return as an ndarray.
            return super(TensorField, self).__getitem__(indices).view(np.ndarray)
        elif has_strs:
            ## Labelled indexing
            # Deal with field portion of index
            has_labels = [index for index in indices if index in self.field_labels]
            has_field = [index for index in indices if index == 'field']
            has_tensor = [index for index in indices if index == 'tensor']

            if not isinstance(indices[0], str):
                raise IndexError("Invalid index")
            if len(has_field) > 1:
                raise IndexError("Invalid index")
            if len(has_tensor) > 1:
                raise IndexError("Invalid index")
            if has_field and has_labels:
                raise IndexError("Invalid index")

            current_key = str(indices[0])
            label_to_ind = {current_key: [1]}
            for ii in range(1, len(indices)):
                index = indices[ii]
                if isinstance(index, str):
                    label_to_ind[current_key].append(ii)
                    label_to_ind[index] = [ii+1]
                    current_key = index
            label_to_ind[current_key].append(len(indices))

            # Deal with tensor portion of index
            indices_tensor = None
            if 'tensor' in label_to_ind:
                ii, jj = label_to_ind.pop('tensor')
                indices_tensor = indices[ii:jj]
            else:
                indices_tensor = (...,)

            # Deal with field portion of index
            indices_field = None
            if 'field' in label_to_ind:
                ii, jj = label_to_ind.pop('field')
                indices_field = indices[ii:jj]
            else:
                indices_field = [slice(None)] * self.field_ndim
                for label in label_to_ind:
                    ii, jj = label_to_ind.pop(label)
                    indices_field[ii] = indices[ii:jj]

            if label_to_ind:
                raise IndexError('Field labels were not found.')

            return self._getitem_tf(tuple(indices_field), tuple(indices_tensor))
        elif isinstance(indices, dict):
            ## Xarray Style Labelled Indexing
            # Dictionary style indexing like that from xarray!
            # Generate a full slice object, then replace it with indices in the dict style indices
            raise NotImplementedError("Didn't make this yet")
        else:
            raise ValueError("Unknown index style.")

    def _getitem_tf(self, field_indices, tensor_indices):
        """Returns a sliced field based on seperate indices for the field and tensor."""

        field_indices = _expand_indices(field_indices, self.field_ndim)
        tensor_indices = _expand_indices(tensor_indices, self.tensor_ndim)

        # Compute updated field attributes
        ii = 0
        field_dx = list()
        field_labels = list()
        for index in field_indices:
            if index is None:
                field_dx.append(None)
                field_labels.append(None)
            elif isinstance(index, int):
                ii += 1
            else:
                field_dx.append(self.field_dx[ii])
                field_labels.append(self.field_labels[ii])
                ii += 1

        # Compute updated tensor attributes
        is_none = [index for index in tensor_indices if index is None]
        is_int = [index for index in tensor_indices if isinstance(index, int)]
        tensor_ndim = self.tensor_ndim + len(is_none) - len(is_int)

        res = super(TensorField, self).__getitem__(field_indices+tensor_indices)
        res._tensor_ndim = tensor_ndim
        res._field_labels = tuple(field_labels)
        res._field_dx = tuple(field_dx)
        return res

    def __setitem__(self, indices, value):
        ## Explicitly perform the data setting using a view of TensorField as
        # an ndarray.
        # See: https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        # TODO: will this even work with advanced indices?
        res = self.__getitem__(indices)
        res.view(np.ndarray)[...] = value
        #np.ndarray.__setitem__(self.view(np.ndarray), indices, value)

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
    def field_labels(self):
        """
        Returns the field grid spacing data.
        """
        return self._field_labels

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
        if self.tensor_ndim == 0:
            return self.shape
        else:
            return self.shape[:-self.tensor_ndim]

    @property
    def tensor_shape(self):
        """
        Returns the shape of the tensor.
        """
        if self.tensor_ndim == 0:
            return ()
        else:
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

    # The following are convenience properties for the common use case of a 3 dimensional vector
    # field
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
