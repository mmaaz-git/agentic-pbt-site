#!/usr/bin/env python3
"""Reproducing the pylong_join count=0 inconsistency bug"""

from Cython.Utility import pylong_join, _pylong_join

# Test the specific failing case
public_result = pylong_join(0)
private_result = _pylong_join(0)

print(f"pylong_join(0) = {repr(public_result)}")
print(f"_pylong_join(0) = {repr(private_result)}")

# Check if they match
if public_result == private_result:
    print("Results match - no bug")
else:
    print(f"Results differ - BUG CONFIRMED: {repr(public_result)} != {repr(private_result)}")

# Test a few more values to understand the pattern
for count in [0, 1, 2, 3]:
    pub = pylong_join(count)
    priv = _pylong_join(count)
    match = "✓" if pub == priv else "✗"
    print(f"count={count}: {match} pylong_join={repr(pub[:50])}... _pylong_join={repr(priv[:50])}...")