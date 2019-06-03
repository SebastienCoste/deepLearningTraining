from mxnet import nd
from mxnet.gluon import nn

x = nd.arange(4)
print(x)
nd.save('x-file', x)

x2 = nd.load('x-file')
print(x2)

y = nd.zeros(4)
nd.save('x-files', [x, y])
x2, y2 = nd.load('x-files')
print((x2, y2))

mydict = {'x': x, 'y': y}
nd.save('mydict', mydict)
mydict2 = nd.load('mydict')
print(mydict2)


class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))


net = MLP()
net.initialize()
x = nd.random.uniform(shape=(2, 20))
y = net(x)

print(y)

net.save_parameters('mlp.params')
clone = MLP()
clone.load_parameters('mlp.params')

yclone = clone(x)
print(y == yclone)








































