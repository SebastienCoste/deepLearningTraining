import mxnet as mx
from mxnet import nd
import numpy as np
from matplotlib import pyplot as plt
from IPython import display
import random
import math

display.set_matplotlib_formats('svg')

probabilities = nd.ones(6) / 6
rand = nd.random.multinomial(probabilities, shape=(5,10))
print(probabilities)
print (rand)

total = 1000
rolls = nd.random.multinomial(probabilities, shape=(total))
counts = nd.zeros((6,total))
totals = nd.zeros(6)
for i, roll in enumerate(rolls):
    totals[int(roll.asscalar())] += 1
    counts[:, i] = totals

print(totals / total)
print(counts)

x = nd.arange(total).reshape((1,total)) + 1
estimates = counts / x
print(estimates[:,0])
print(estimates[:,1])
print(estimates[:,2])
print(estimates[:,100])

# plt.figure(figsize=(8, 6))
# for i in range(6):
#     plt.plot(estimates[i, :].asnumpy(), label=("P(die=" + str(i) +")"))
#
# plt.axhline(y=0.16666, color='black', linestyle='dashed')
# plt.legend()
# plt.show()


for i in range(10):
    print(random.random())


counts = np.zeros(100)
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
axes = axes.reshape(6)
# Mangle subplots such that we can index them in a linear fashion rather than
# a 2D grid
# for i in range(1, 1000001):
#     counts[random.randint(0, 99)] += 1
#     if i in [10, 100, 1000, 10000, 100000, 1000000]:
#         axes[int(math.log10(i))-1].bar(np.arange(1, 101), counts)
# plt.show()


# Number of samples
n = 1000000
y = np.random.uniform(0, 1, n)
x = np.arange(1, n+1)
# Count number of occurrences and divide by the number of total draws
p0 = np.cumsum(y < 0.35) / x
p1 = np.cumsum(y >= 0.35) / x

# plt.figure(figsize=(15, 8))
# plt.semilogx(x, p0)
# plt.semilogx(x, p1)
# plt.show()

x = np.arange(-10, 10, 0.01)
p = (1/math.sqrt(2 * math.pi)) * np.exp(-0.5 * x**2)
# plt.figure(figsize=(10, 5))
# plt.plot(x, p)
# plt.show()

# Generate 10 random sequences of 10,000 uniformly distributed random variables
tmp = np.random.uniform(size=(10000,10))
x = 1.0 * (tmp > 0.3) + 1.0 * (tmp > 0.8)
mean = 1 * 0.5 + 2 * 0.2
variance = 1 * 0.5 + 4 * 0.2 - mean**2
print('mean {}, variance {}'.format(mean, variance))

# Cumulative sum and normalization
y = np.arange(1,10001).reshape(10000,1)
z = np.cumsum(x,axis=0) / y
print(z)

plt.figure(figsize=(10,5))
for i in range(10):
    plt.semilogx(y,z[:,i])

plt.semilogx(y,(variance**0.5) * np.power(y,-0.5) + mean,'r')
plt.semilogx(y,-(variance**0.5) * np.power(y,-0.5) + mean,'r')
plt.show()










