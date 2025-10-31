import numpy as np
from scipy.odr import Data, Model, ODR


def linear_func(beta, x):
    return beta[0] * x + beta[1]


x = np.array([0., 2.5, 5., 7.5, 10.])
y = np.array([2.225e-311, 2.5, 5.0, 7.5, 10.0])

data = Data(x, y)
model = Model(linear_func)

initial_guess = [0.9, 2.003e-311]

odr_obj = ODR(data, model, beta0=initial_guess)
result = odr_obj.run()

print(f"Result beta: {result.beta}")
print(f"Has NaN: {np.any(np.isnan(result.beta))}")


print("\nWith sanitized initial guess (replacing subnormal with 0):")
odr_obj2 = ODR(data, model, beta0=[0.9, 0.0])
result2 = odr_obj2.run()
print(f"Result beta: {result2.beta}")
print(f"Has NaN: {np.any(np.isnan(result2.beta))}")