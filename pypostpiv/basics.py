"""A set of functions for basic field operations
"""
import warnings
import numpy as np

def fsum(field, axis):
    return np.nansum(field, axis=0, keepdims=True)

def mag(field):
    """Compute the magnitude of the field

    Author(s)
    ---------
    Jia Cheng Hu
    """
    return np.sqrt(np.sum(field**2, axis=0, keepdims=True))

def fmean(field):
    """Compute the mean of the field

    Author(s)
    ---------
    Jia Cheng Hu
    """
    warnings.simplefilter("ignore")
    return np.nanmean(field, axis=3, keepdims=True)

def rms(field):
    """Compute the root mean square of the field

    Author(s)
    ---------
    Jia Cheng Hu
    """
    return np.nanstd(field, axis=3, keepdims=True)

def ddx(field, method):
    """Compute the derivative of field with respect to x-axis

    Method
    ---------
    'central'      - central difference scheme, second order accuracy (default)
    'richardson'   - Richardson extrapolation scheme, third order accuracy
    'least square' - least square scheme, second order accuracy

    Author(s)
    ---------
    Jia Cheng Hu
    """
    if method == None or method == 'central':
        new_field = (field[:,2:,1:-1]-field[:,:-2,1:-1])/field.dL/2
        new_field.x = field.x[1:-1,1:-1]
        new_field.y = field.y[1:-1,1:-1]
        return new_field

    elif method == 'richardson':
        new_field = field[:,:-4,2:-2] - 8*field[:,1:-3,2:-2]  + 8*field[:,3:-1,2:-2]  -  field[:,4:,2:-2]
        new_field = new_field/field.dL/12
        new_field.x = field.x[2:-2,2:-2]
        new_field.y = field.y[2:-2,2:-2]
        return new_field

    elif method == 'least_square':
        new_field = 2*field[:,4:,2:-2] + field[:,3:-1,2:-2] - field[:,1:-3,2:-2] - 2*field[:,:-4,2:-2]
        new_field = new_field/field.dL/10
        new_field.x = field.x[2:-2,2:-2]
        new_field.y = field.y[2:-2,2:-2]
        return new_field

    else:
        assert()

def ddy(field, method):
    """Compute the derivative of field with respect to y-axis

    Method
    ---------
    'central'      - central difference scheme, second order accuracy (default)
    'richardson'   - Richardson extrapolation scheme, third order accuracy
    'least square' - least square scheme, second order accuracy

    Author(s)
    ---------
    Jia Cheng Hu
    """
    if method == None or method == 'central':
        new_field = (field[:,1:-1,:-2] - field[:,1:-1,2:])/field.dL/2
        new_field.x = field.x[1:-1,1:-1]
        new_field.y = field.y[1:-1,1:-1]
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
        assert()
