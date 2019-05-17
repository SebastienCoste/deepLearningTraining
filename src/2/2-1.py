import mxnet as mx
from mxnet import nd
import numpy as np

x = nd.arange(12).reshape((3,4))
y = nd.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
z = nd.arange(9).reshape((3, 3))
init_z = id(z)
verif = nd.dot(x, y.T) + z
print(x)
print(y)
print(z)

nd.elemwise_add(z, nd.dot(x, y.T), out=z)
late_z = id(z)
print (verif == z)
print ("z didn't move?", init_z == late_z)
