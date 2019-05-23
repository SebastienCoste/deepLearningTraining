from mxnet import gluon, nd
from mxnet.gluon import nn

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()


layer = CenteredLayer()
print(layer(nd.array([1, 2, 3, 4, 5])))

net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
y = net(nd.random.uniform(shape=(4, 8)))
print(y.mean().asscalar())

















































