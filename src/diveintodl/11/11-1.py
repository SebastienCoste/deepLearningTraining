from mxnet import nd, sym
from mxnet.gluon import nn
import time
import random

def get_net():
    net = nn.HybridSequential()  # Here we use the class HybridSequential
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net

x = nd.random.normal(shape=(1, 512))
# net = get_net()
# print(net(x))
#
# net.hybridize()
# print(net(x))


def benchmark(net, x):
    start = time.time()
    for i in range(1000):
        _ = net(x)
    # To facilitate timing, we wait for all computations to be completed
    nd.waitall()
    return time.time() - start

# net = get_net()
# print('before hybridizing: %.4f sec' % (benchmark(net, x)))
# net.hybridize()
# print('after hybridizing: %.4f sec' % (benchmark(net, x)))


# net.export('my_mlp')


class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(10)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x):
        # x.asnumpy()
        if random.random() >0:
            print("ok")
        print('F: ', F)
        print('x: ', x)
        x = F.relu(self.hidden(x))
        print('hidden: ', x)
        return self.output(x)


net = HybridNet()
net.initialize()
x = nd.random.normal(shape=(1, 4))
print(net(x))
net.hybridize()
print(net(x))
print(net(x))










































