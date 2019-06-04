from mxnet import nd
from mxnet.gluon import nn

# For convenience, we define a function to calculate the convolutional layer.
# This function initializes the convolutional layer weights and performs
# corresponding dimensionality elevations and reductions on the input and
# output
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # (1,1) indicates that the batch size and the number of channels
    # (described in later chapters) are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Exclude the first two dimensions that do not interest us: batch and
    # channel
    return Y.reshape(Y.shape[2:])

# Note that here 1 row or column is padded on either side, so a total of 2
# rows or columns are added
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = nd.random.uniform(shape=(8, 8))
R = comp_conv2d(conv2d, X).shape
print(R)


conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
R = comp_conv2d(conv2d, X).shape
print(R)

conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
R= comp_conv2d(conv2d, X).shape
print(R)

conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
R = comp_conv2d(conv2d, X).shape
print(R)















































