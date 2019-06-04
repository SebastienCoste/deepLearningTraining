from mxnet import nd
from mxnet.gluon import nn

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = nd.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y


X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
R = pool2d(X, (2, 2))
print(R)
R = pool2d(X, (2, 2), 'avg')
print(R)

X = nd.arange(16).reshape((1, 1, 4, 4))
print(X)
pool2d = nn.MaxPool2D(3)
# Because there are no model parameters in the pooling layer, we do not need
# to call the parameter initialization function
R = pool2d(X)
print(R)
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
R = pool2d(X)
print(R)

# X = nd.concat(X, X + 1, dim=0)
print(nd.concat(X, X + 1, dim=0))
print(nd.concat(X, X + 1, dim=1))
print(nd.concat(X, X + 1, dim=2))
print(nd.concat(X, X + 1, dim=3))











