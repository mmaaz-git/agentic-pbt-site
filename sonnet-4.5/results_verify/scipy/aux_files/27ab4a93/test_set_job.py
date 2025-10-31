import numpy as np
from scipy.odr import Data, Model, ODR


def fcn(B, x):
    return B[0] * x + B[1]


model = Model(fcn)
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
data = Data(x, y)
odr_obj = ODR(data, model, beta0=[1.0, 0.0])

print("Testing set_job with fit_type=5 (invalid value)...")
try:
    odr_obj.set_job(fit_type=5)
    print("No error raised - invalid value was silently ignored/handled")
except Exception as e:
    print(f"Error raised: {e}")

print("\nTesting set_job with deriv=5 (invalid value)...")
try:
    odr_obj.set_job(deriv=5)
    print("No error raised - invalid value was silently ignored/handled")
except Exception as e:
    print(f"Error raised: {e}")

print("\nChecking job value after setting invalid parameters...")
print(f"job value: {odr_obj.job}")