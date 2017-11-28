"""
A set of functions for basic field operations
"""

import warnings
import numpy as np

def fsum(field, axis):
    return np.nansum(field, axis=0, keepdims=True)

def mag(field):
    """
    Compute the magnitude of the field.

    Parameters
    ----------
    field : Field2d

    Author(s)
    ---------
    Jia Cheng Hu
    """
    return np.sqrt(np.sum(field**2, axis=0, keepdims=True))

def fmean(field):
    """
    Compute the time mean of the field.

    Parameters
    ----------
    field : Field2d

    Author(s)
    ---------
    Jia Cheng Hu
    """
    warnings.simplefilter("ignore")
    return np.nanmean(field, axis=3, keepdims=True)

def rms(field):
    """
    Compute the root mean square of the field.

    Parameters
    ----------
    field : Field2d

    Author(s)
    ---------
    Jia Cheng Hu
    """
    return np.nanstd(field, axis=3, keepdims=True)

def ddx(field, method=None):
    """
    Compute the derivative of a field with respect to the x-axis.

    Parameters
    ----------
    field : Field2d
    method : string, optional
        Must be one of:
        'central' - central difference scheme, second order accuracy (default)
        'richardson' - Richardson extrapolation scheme, third order accuracy
        'least square' - least square scheme, second order accuracy

    Returns
    -------
    Field2d

    Author(s)
    ---------
    Jia Cheng Hu
    Jonathan Deng
    """
    if method == None or method == 'central':
        new_field = field-field

        # Apply central differencing in the 'core' region
        new_field[:,1:-1] = (field[:,2:]-field[:,:-2])/field.dL/2

        # Apply second order forward/backward differences at boundaries
        new_field[:,0] = (field[:,2] - 2*field[:,1] + field[:,0]) / \
                         field.dL**2
        new_field[:,-1] = (field[:,-3] - 2*field[:,-2] + field[:,-1]) / \
                          field.dL**2
        return new_field

    elif method == 'richardson':
        new_field = field[:,:-4,2:-2] - field[:,4:,2:-2] + \
                    8*field[:,3:-1,2:-2] - 8*field[:,1:-3,2:-2]
        new_field = new_field/field.dL/12
        new_field.x = field.x[2:-2,2:-2]
        new_field.y = field.y[2:-2,2:-2]
        return new_field

    elif method == 'least_square':
        new_field = 2*field[:,4:,2:-2] - 2*field[:,:-4,2:-2] + \
                    field[:,3:-1,2:-2] - field[:,1:-3,2:-2]
        new_field = new_field/field.dL/10
        new_field.x = field.x[2:-2,2:-2]
        new_field.y = field.y[2:-2,2:-2]
        return new_field

    else:
        raise ValueError('method keyword argument was not valid.')

def ddy(field, method=None):
    """
    Compute the derivative of a field with respect to the y-axis.

    Parameters
    ----------
    field : Field2d
    method : string, optional
        Must be one of:
        'central' - central difference scheme, second order accuracy (default)
        'richardson' - Richardson extrapolation scheme, third order accuracy
        'least square' - least square scheme, second order accuracy

    Returns
    -------
    Field2d

    Author(s)
    ---------
    Jia Cheng Hu
    Jonathan Deng
    """
    if method == None or method == 'central':
        new_field = field-field

        # Apply central differencing in the 'core' region
        new_field[:,:,1:-1] = (field[:,:,2:]-field[:,:,:-2])/field.dL/2

        # Apply second order forward/backward differences at boundaries
        new_field[:,:,0] = (field[:,:,2] - 2*field[:,:,1] + field[:,:,0]) / \
                         field.dL**2
        new_field[:,:,-1] = (field[:,:,-3] - 2*field[:,:,-2] + field[:,:,-1]) / \
                          field.dL**2
        return new_field

    elif method == 'richardson':
        new_field = field[:,2:-2,4:] - 8*field[:,2:-2,3:-1] + 8*field[:,2:-2,1:-3] - field[:,2:-2,:-4]
        new_field = new_field/field.dL/12
        new_field.x = field.x[2:-2,2:-2]
        new_field.y = field.y[2:-2,2:-2]
        return new_field

    elif method == 'least_square':
        new_field = 2*field[:,2:-2,:-4] + field[:,2:-2,1:-3] - field[:,2:-2,3:-1] - 2*field[:,2:-2,4:]
        new_field = new_field/field.dL/10
        new_field.x = field.x[2:-2,2:-2]
        new_field.y = field.y[2:-2,2:-2]
        return new_field

    else:
        raise ValueError('method keyword argument was not valid.')
