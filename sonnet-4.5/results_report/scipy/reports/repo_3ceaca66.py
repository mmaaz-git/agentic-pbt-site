import numpy as np
from scipy.odr import Data, Model, ODR


def linear_func(beta, x):
    return beta[0] * x + beta[1]


# Create data with very small intercept (subnormal float)
x = np.array([0., 2.5, 5., 7.5, 10.])
y = np.array([2.225e-311, 2.5, 5.0, 7.5, 10.0])

data = Data(x, y)
model = Model(linear_func)

# Initial guess with subnormal float
initial_guess = [0.9, 2.003e-311]

print("Running ODR with subnormal float in initial guess...")
print(f"Initial guess: {initial_guess}")
print(f"Is initial guess[1] subnormal? {abs(initial_guess[1]) < np.finfo(np.float64).tiny and initial_guess[1] != 0}")

odr_obj = ODR(data, model, beta0=initial_guess)
result = odr_obj.run()

print(f"\nResult beta: {result.beta}")
print(f"Has NaN: {np.any(np.isnan(result.beta))}")

print("\n" + "="*50)
print("With sanitized initial guess (replacing subnormal with 0):")
initial_guess_sanitized = [0.9, 0.0]
print(f"Initial guess: {initial_guess_sanitized}")

odr_obj2 = ODR(data, model, beta0=initial_guess_sanitized)
result2 = odr_obj2.run()
print(f"\nResult beta: {result2.beta}")
print(f"Has NaN: {np.any(np.isnan(result2.beta))}")