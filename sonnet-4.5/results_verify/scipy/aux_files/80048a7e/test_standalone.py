import numpy as np
import scipy.odr as odr

n = 5
x = np.linspace(0, 10, n)
y = 2 * x + 1

def linear_func(B, x):
    return B[0] * x + B[1]

model = odr.Model(linear_func)
data = odr.Data(x, y)
delta0 = np.zeros(n)

odr_obj = odr.ODR(data, model, beta0=[1.0, 0.0], delta0=delta0)
output = odr_obj.run()