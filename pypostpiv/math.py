"""
A set of functions for basic field operations
"""

import warnings
import numpy as np

def fsum(field):
    return np.nansum(field, axis=-1, keepdims=True)

def mag(field):
    """
    Compute the magnitude of the field.

    Parameters
    ----------
    field : TensorField

    Author(s)
    ---------
    Jia Cheng Hu
    """
    return np.sqrt(np.sum(field**2, axis=-1))

def fmean(field):
    """
    Compute the time mean of the field.

    Parameters
    ----------
    field : TensorField

    Author(s)
    ---------
    Jia Cheng Hu
    """
    warnings.simplefilter("ignore")
    return np.nanmean(field, axis=3)

def rms(field):
    """
    Compute the root mean square of the field.

    Parameters
    ----------
    field : TensorField

    Author(s)
    ---------
    Jia Cheng Hu
    """
    return np.nanstd(field, axis=3, keepdims=True)

def ddx(field, method='central'):
    """
    Compute the derivative of a field with respect to the x-axis.

    Parameters
    ----------
    field : TensorField
    method : string, optional
        Must be one of:
        'central' - central difference scheme, second order accuracy (default)
        'richardson' - Richardson extrapolation scheme, third order accuracy
        'lsq' - least square scheme, second order accuracy

    Returns
    -------
    Field2d

    Author(s)
    ---------
    Jia Cheng Hu
    """
    new_field = None
    if method == 'central':
        new_field = field.copy()

        # Apply central differencing in the 'core' region
        new_field[1:-1] = (field[2:]-field[:-2])/field.dx/2

        # Apply second order forward/backward differences at boundaries
        new_field[0] = (field[2] - 2*field[1] + field[0]) / field.dx**2
        new_field[-1] = (field[-3] - 2*field[-2] + field[-1]) / field.dx**2
    elif method == 'richardson':
        new_field = field[:-4, 2:-2] - field[4:, 2:-2] + \
                    8*field[3:-1, 2:-2] - 8*field[1:-3, 2:-2]
        new_field = new_field/field.dx/12
    elif method == 'lsq':
        new_field = 2*field[4:, 2:-2] - 2*field[:-4, 2:-2] + \
                    field[3:-1, 2:-2] - field[1:-3, 2:-2]
        new_field = new_field/field.dx/10
    else:
        raise ValueError("'method' must be one of 'central', 'richardson', or 'lsq'")

    return new_field

def ddy(field, method='central'):
    """
    Compute the derivative of a field with respect to the y-axis.

    Parameters
    ----------
    field : TensorField
    method : string, optional
        Must be one of:
        'central' - central difference scheme, second order accuracy (default)
        'richardson' - Richardson extrapolation scheme, third order accuracy
        'lsq' - least square scheme, second order accuracy

    Returns
    -------
    TensorField

    Author(s)
    ---------
    Jia Cheng Hu
    """
    new_field = None
    if method == 'central':
        new_field = field.copy()

        # Apply central differencing in the 'core' region
        new_field[:, 1:-1] = (field[:, 2:]-field[:, :-2])/field.dy/2

        # Apply second order forward/backward differences at boundaries
        new_field[:, 0] = (field[:, 2] - 2*field[:, 1] + field[:, 0]) / field.dy**2
        new_field[:, -1] = (field[:, -3] - 2*field[:, -2] + field[:, -1]) / field.dy**2
    elif method == 'richardson':
        new_field = field[2:-2, 4:] - 8*field[2:-2, 3:-1] + 8*field[2:-2, 1:-3] - field[2:-2, :-4]
        new_field = new_field/field.dy/12
    elif method == 'lsq':
        new_field = 2*field[2:-2, :-4] + field[2:-2, 1:-3] - field[2:-2, 3:-1] - 2*field[2:-2, 4:]
        new_field = new_field/field.dy/10
    else:
        raise ValueError("'method' must be one of 'central', 'richardson', or 'lsq'")
    return new_field
