"""
Tests of the piv module
"""

import sys

import numpy as np

sys.path.append('..')
from pypostpiv import piv

fielda = piv.TensorField((4, 5, 6), (3, ))
fieldb = piv.TensorField((4, 5, 1), (1, ))

arraya = np.zeros((4, 5, 6, 3))
arrayb = np.zeros((4, 5, 6, 3))

fieldc = fielda + fieldb

print("Testing binary ufuncs on TensorField / TensorField dtypes")
res = fielda + fieldb
print(f"{type(fielda)} + {type(fieldb)} = {type(res)}")
print(f"{fielda.field_dx} + {fieldb.field_dx} -> {res.field_dx}")

print("\nTesting binary ufuncs on TensorField / np.ndarray dtypes")
res = arraya + fieldb
print(f"{type(arraya)} + {type(fieldb)} -> {type(res)}")
print(f"{type(arraya)} + {fieldb.field_dx} -> {res.field_dx}")

print("\nTesting binary ufuncs on np.ndarray / TensorField dtypes")
res = fielda + arrayb
print(f"{type(fielda)} + {type(arrayb)} -> {type(res)}")
print(f"{fielda.field_dx} + {type(arrayb)} -> {res.field_dx}")

#import ipdb; ipdb.set_trace()
test = fielda[..., 0]
fielda[..., 0] = 123

print(f"repr(fielda[:, 0, :, ...]) >>> {repr(fielda[:, 0, :, ...])}")

#np.ravel(fielda)
#np.reshape(fielda, (20, 18))
#np.squeeze(fielda)

print("\nTesting sum")
res = np.sum(fielda, axis=1)
print(f"type(np.cumsum(fielda, axis=0)) -> {type(res)}")
print(f"np.cumsum(fielda, axis=0).dx -> {res.field_dx}")
print(f"np.cumsum(fielda, axis=0).shape -> {res.shape}")
