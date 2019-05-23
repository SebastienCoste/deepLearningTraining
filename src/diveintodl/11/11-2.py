from mxnet import autograd, gluon, nd
from mxnet.gluon import loss as gloss, nn
import os
import subprocess
import time


# This class has been saved in the Gluonbook module for future use
class Benchmark():
    def __init__(self, prefix=None):
        self.prefix = prefix + ' ' if prefix else ''

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *args):
        print('%stime: %.4f sec' % (self.prefix, time.time() - self.start))


with Benchmark('Workloads are queued.'):
    x = nd.random.uniform(shape=(2000, 2000))
    y = nd.dot(x, x).sum()

with Benchmark('Workloads are finished.'):
    print('sum =', y)


with Benchmark("wait_to_read"):
    y = nd.dot(x, x)
    y.wait_to_read()


with Benchmark("waitall"):
    y = nd.dot(x, x)
    z = nd.dot(x, x)
    nd.waitall()

with Benchmark("asnumpy"):
    y = nd.dot(x, x)
    y.asnumpy()

with Benchmark("norm.asscalar"):
    y = nd.dot(x, x)
    y.norm().asscalar()

with Benchmark("norm"):
    y = nd.dot(x, x)
    y.norm()

# with Benchmark('synchronous.'):
#     for _ in range(1000):
#         y = x + 1
#         y.wait_to_read()
#
# with Benchmark('asynchronous.'):
#     for _ in range(1000):
#         y = x + 1
#     nd.waitall()


def data_iter():
    start = time.time()
    num_batches, batch_size = 100, 1024
    for i in range(num_batches):
        X = nd.random.normal(shape=(batch_size, 512))
        y = nd.ones((batch_size,))
        yield X, y
        if (i + 1) % 50 == 0:
            print('batch %d, time %f sec' % (i + 1, time.time() - start))


def get_mem():
    res = subprocess.check_output(['ps', 'u', '-p', str(os.getpid())])
    return int(str(res).split()[15]) / 1e3


net = nn.Sequential()
net.add(nn.Dense(2048, activation='relu'),
        nn.Dense(512, activation='relu'),
        nn.Dense(1))
net.initialize()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.005})
loss = gloss.L2Loss()

for X, y in data_iter():
    break
loss(y, net(X)).wait_to_read()


l_sum, mem = 0, get_mem()
for X, y in data_iter():
    with autograd.record():
        l = loss(y, net(X))
    # Use of the Asscalar synchronization function
    l_sum += l.mean().asscalar()
    l.backward()
    trainer.step(X.shape[0])
nd.waitall()
print('increased memory: %f MB' % (get_mem() - mem))


mem = get_mem()
for X, y in data_iter():
    with autograd.record():
        l = loss(y, net(X))
    l.backward()
    trainer.step(X.shape[0])
nd.waitall()
print('increased memory: %f MB' % (get_mem() - mem))






































