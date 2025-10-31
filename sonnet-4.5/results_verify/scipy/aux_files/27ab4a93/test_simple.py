import numpy as np
from scipy.odr import Data, Model, ODR


def fcn(B, x):
    return B[0] * x + B[1]


model = Model(fcn)
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
data = Data(x, y)
odr_obj = ODR(data, model, beta0=[1.0, 0.0], rptfile="test.txt")

print("Testing set_iprint with init=3 (invalid value)...")
try:
    odr_obj.set_iprint(init=3)
    print("No error raised!")
except ValueError as e:
    print(f"ValueError raised: {e}")

print("\nTesting set_iprint with init=1 (valid value)...")
try:
    odr_obj.set_iprint(init=1)
    print("Success - no error raised for valid value")
except ValueError as e:
    print(f"Unexpected error: {e}")

print("\nTesting set_iprint with init=5, so_init=10 (both invalid)...")
try:
    odr_obj.set_iprint(init=5, so_init=10)
    print("No error raised!")
except ValueError as e:
    print(f"ValueError raised: {e}")