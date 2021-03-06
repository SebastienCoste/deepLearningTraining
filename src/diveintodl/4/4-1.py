import sys
sys.path.insert(0, '..')

# %matplotlib inline
import d2l
from mxnet import autograd, nd

def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.asnumpy(), y_vals.asnumpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')
    d2l.plt.show()


x = nd.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    # y = x.relu()
    # y = x.tanh()
    y = x.sigmoid()
xyplot(x, y, 'relu')

y.backward()
# xyplot(x, x.grad, 'grad of relu')