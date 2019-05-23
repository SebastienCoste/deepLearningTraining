from mxnet import init, nd
from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()  # Use the default initialization method

x = nd.random.uniform(shape=(2, 20))
print(net(x))  # Forward computation
print(net[0].params)
print(net[1].params)
print(net[1].bias)
print(net[1].bias.data())
print(net[0].params['dense0_weight'])
print(net[0].params['dense0_weight'].data())
print(net[0].weight.grad())

# parameters only for the first layer
print(net[0].collect_params())
# parameters of the entire network
print(net.collect_params())

print(net.collect_params()['dense1_bias'].data())


def block1():
    net = nn.Sequential()
    net.add(nn.Dense(32, activation='relu'))
    net.add(nn.Dense(16, activation='relu'))
    return net


def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add(block1())
    return net

rgnet = nn.Sequential()
rgnet.add(block2())
rgnet.add(nn.Dense(10))
rgnet.initialize()
rgnet(x)


print(rgnet.collect_params)
print(rgnet.collect_params())

# force_reinit ensures that the variables are initialized again, regardless of
# whether they were already initialized previously
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
print(net[0].weight.data()[0])

net.initialize(init=init.Constant(1), force_reinit=True)
print(net[0].weight.data()[0])


net = nn.Sequential()
# We need to give the shared layer a name such that we can reference its
# parameters
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

x = nd.random.uniform(shape=(2, 20))
net(x)

# Check whether the parameters are the same
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0,0] = 100
# Make sure that they're actually the same object rather than just having the
# same value
print(net[1].weight.data()[0] == net[2].weight.data()[0])

































