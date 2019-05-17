
from mxnet import nd

A = nd.arange(20).reshape((5,4))
# print(A)
# print(A.T)
# print(nd.sum(A))
# print(nd.mean(A))
# print(nd.sum(A) / A.size)

a = 2
x = nd.ones(3)
y = nd.zeros(3)
# print(x.shape)
# print(y.shape)
# print((a * x).shape)
# print((a * x + y).shape)
#
# print(x)
# print(nd.sum(x))

x = nd.arange(4)
y = nd.ones(4)
# print(x, y, nd.dot(x, y))
# print ((x.T * y).sum())


A = nd.arange(20).reshape((4,4))
B = nd.ones(shape=(4, 2))
C = nd.dot(A, B)
print (A)
print (B)
print (C)