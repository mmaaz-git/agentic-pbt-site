from scipy.odr import ODR, Data, Model

def linear_func(B, x):
    return B[0] * x + B[1]

data = Data([1, 2, 3], [2, 4, 6])
model = Model(linear_func)
odr = ODR(data, model, beta0=[1, 1])

# This should crash with ValueError
odr.iprint = 3
odr.set_iprint(final=0)