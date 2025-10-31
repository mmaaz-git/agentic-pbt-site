"""
Demonstrating the crash of pandas.api.types.infer_dtype with Python scalar types.
"""

import pandas.api.types as types
import numpy as np

print("Testing pandas.api.types.infer_dtype with various scalar types\n")
print("=" * 60)

# Test cases that should work according to documentation but crash
test_cases_crash = [
    ("Python int", 0),
    ("Python float", 1.5),
    ("Python bool", True),
    ("Python complex", 1+2j),
    ("None", None)
]

print("\n1. PYTHON BUILT-IN SCALARS (These crash):\n")
for name, value in test_cases_crash:
    try:
        result = types.infer_dtype(value, skipna=False)
        print(f"{name:20} value={str(value):10} → {result}")
    except TypeError as e:
        print(f"{name:20} value={str(value):10} → ERROR: {e}")

print("\n" + "=" * 60)
print("\n2. SCALAR TYPES THAT WORK:\n")

# Test cases that work
test_cases_work = [
    ("String", "hello"),
    ("Bytes", b"bytes"),
    ("NumPy int64", np.int64(5)),
    ("NumPy float64", np.float64(5.5))
]

for name, value in test_cases_work:
    try:
        result = types.infer_dtype(value, skipna=False)
        print(f"{name:20} value={value!r:15} → {result}")
    except TypeError as e:
        print(f"{name:20} value={value!r:15} → ERROR: {e}")

print("\n" + "=" * 60)
print("\n3. THE SAME VALUES WRAPPED IN LISTS (All work):\n")

# Test that same values work when wrapped in lists
for name, value in test_cases_crash:
    try:
        result = types.infer_dtype([value], skipna=False)
        print(f"{name:20} value=[{value}] → {result}")
    except TypeError as e:
        print(f"{name:20} value=[{value}] → ERROR: {e}")

print("\n" + "=" * 60)
print("\n4. VERIFYING SCALAR STATUS WITH pandas.api.types.is_scalar:\n")

# Verify that pandas considers these as scalars
for name, value in test_cases_crash + test_cases_work:
    is_scalar = types.is_scalar(value)
    print(f"{name:20} is_scalar={is_scalar}")

print("\n" + "=" * 60)
print("\nCONCLUSION:")
print("The function is documented to accept 'scalar' values but crashes on")
print("common Python scalar types (int, float, bool, complex, None) while")
print("working for some other scalars (str, bytes, numpy scalars).")