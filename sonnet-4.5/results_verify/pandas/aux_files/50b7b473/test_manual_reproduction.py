#!/usr/bin/env python3
"""Manual reproduction test from the bug report"""

from pandas.plotting._matplotlib.converter import TimeFormatter

formatter = TimeFormatter(locs=[])
x = 86400.99999999997

print(f"Testing x = {x}")
print(f"int(x) = {int(x)}")
print(f"(x - int(x)) * 10**6 = {(x - int(x)) * 10**6}")
print(f"round((x - int(x)) * 10**6) = {round((x - int(x)) * 10**6)}")

try:
    result = formatter(x)
    print(f"Success: formatter({x}) = {result}")
except ValueError as e:
    print(f"ValueError: {e}")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")