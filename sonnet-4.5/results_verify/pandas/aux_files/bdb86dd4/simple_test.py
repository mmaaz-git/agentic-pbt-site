import pandas as pd

print("Testing 1 ** -1 with IntegerArray...")
base = pd.array([1], dtype="Int64")
exponent = pd.array([-1], dtype="Int64")

try:
    result = base ** exponent
    print(f"Result: {result}")
except ValueError as e:
    print(f"ValueError raised: {e}")

print("\nTesting regular numpy behavior for 1 ** -1:")
import numpy as np
# Regular numpy raises error for int ** negative int
try:
    result = np.array([1], dtype=np.int64) ** np.array([-1], dtype=np.int64)
    print(f"Numpy result: {result}")
except ValueError as e:
    print(f"Numpy also raises ValueError: {e}")

print("\nTesting float version (1.0 ** -1):")
result_float = np.array([1.0]) ** np.array([-1])
print(f"Float result: {result_float}")

print("\nMathematically, 1 ** -1 = 1")
print("Since 1 ** -1 = 1 / (1 ** 1) = 1 / 1 = 1")