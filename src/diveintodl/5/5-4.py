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

params = gluon.ParameterDict()
params.get('param2', shape=(2, 3))
params


class MyDense(nn.Block):
    # units: the number of outputs in this layer; in_units: the number of
    # inputs in this layer
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)











































