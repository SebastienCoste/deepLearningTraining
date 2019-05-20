from mxnet import autograd, nd

# x = nd.arange(4).reshape((4, 1))
# print(x)
# x.attach_grad()
# with autograd.record():
#     y = 2 * nd.dot(x.T, x)
# print(y)
# z = y.backward()
# print((x.grad - 4 * x).norm().asscalar() == 0)
# print(x.grad)
#
# print(autograd.is_training())
# with autograd.record():
#     print(autograd.is_training())

# def f(a):
#     b = a * 2
#     while b.norm().asscalar() < 1000:
#         b = b * 2
#     if b.sum().asscalar() > 0:
#         c = b
#     else:
#         c = 100 * b
#     return c
#
# a = nd.random.normal(scale = 2, shape=3)
# print("a= ", a)
# a.attach_grad()
# with autograd.record():
#     d = f(a)
# d.backward()
# print("d = ", d)
# print("a.grad == (d / a) ?", a.grad == (d / a))
# print("grad= ", a.grad )
# print("d / a = ", d / a )


elim = nd.zeros(10)
elim[8] = 1


def second_bid(a):
    b = a.copy()
    s = b * elim
    z = s
    print("res= ", z)
    return z


x = nd.sort(nd.random.randn(1, 10, loc=20, scale=5)[0])
print("init x = ", x)
x.attach_grad()
with autograd.record():
    d = second_bid(x)
d.backward()
print("grad= ", x.grad)






