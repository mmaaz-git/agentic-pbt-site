#!/usr/bin/env python3
"""Reproduce the reported bug in scipy.odr.ODR.set_iprint"""
import numpy as np
from scipy import odr

# First, try the exact reproduction case from the bug report
print("=== Test 1: Exact reproduction case ===")
try:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    data = odr.Data(x, y)
    model = odr.unilinear
    odr_obj = odr.ODR(data, model, beta0=[1.0, 1.0], rptfile='test_report.txt')

    print(f"Setting init=0, so_init=1...")
    odr_obj.set_iprint(init=0, so_init=1)
    print("Success! No error occurred")
except ValueError as e:
    print(f"ValueError occurred: {e}")
except Exception as e:
    print(f"Other exception occurred: {e.__class__.__name__}: {e}")

print("\n=== Test 2: Test without rptfile ===")
try:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 2.0])
    data = odr.Data(x, y)
    model = odr.unilinear
    odr_obj = odr.ODR(data, model, beta0=[1.0, 1.0])  # No rptfile specified

    print(f"Setting init=0, so_init=1 with no rptfile...")
    odr_obj.set_iprint(init=0, so_init=1)
    print("Success! No error occurred")
except ValueError as e:
    print(f"ValueError occurred: {e}")
except odr.OdrError as e:
    print(f"OdrError occurred: {e}")
except Exception as e:
    print(f"Other exception occurred: {e.__class__.__name__}: {e}")

print("\n=== Test 3: Test other combinations ===")
test_cases = [
    (0, 0),  # Should work (no output)
    (1, 0),  # Should work (short to file only)
    (2, 0),  # Should work (long to file only)
    (0, 1),  # Bug report says this fails
    (0, 2),  # Bug report says this would also fail
    (1, 1),  # Should work
    (2, 2),  # Should work
]

for init_val, so_init_val in test_cases:
    try:
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0, 2.0])
        data = odr.Data(x, y)
        model = odr.unilinear
        odr_obj = odr.ODR(data, model, beta0=[1.0, 1.0], rptfile='test.txt')

        print(f"Setting init={init_val}, so_init={so_init_val}... ", end="")
        odr_obj.set_iprint(init=init_val, so_init=so_init_val)
        print("OK")
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"ERROR: {e.__class__.__name__}: {e}")