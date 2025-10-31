#!/usr/bin/env python3
"""Test practical impacts of the bug"""

from xarray.core.dtypes import AlwaysGreaterThan, AlwaysLessThan, INF, NINF
import numpy as np

print("=== Testing sorting behavior ===")
# Test with AlwaysGreaterThan
values = [1, INF, 5, INF, 3]
print(f"Original list: {values}")
try:
    sorted_values = sorted(values)
    print(f"Sorted list: {sorted_values}")
except Exception as e:
    print(f"Error sorting: {e}")

# Test with mixed values
values2 = [NINF, 1, INF, 0, -1]
print(f"\nOriginal list: {values2}")
try:
    sorted_values2 = sorted(values2)
    print(f"Sorted list: {sorted_values2}")
except Exception as e:
    print(f"Error sorting: {e}")

print("\n=== Testing with numpy inf for comparison ===")
values3 = [-np.inf, 1, np.inf, 0, -1]
print(f"Original list with np.inf: {values3}")
sorted_values3 = sorted(values3)
print(f"Sorted list: {sorted_values3}")

print("\n=== Testing max/min functions ===")
# What happens with max/min?
values4 = [1, 2, INF, 3]
print(f"List: {values4}")
print(f"max(): {max(values4)}")
print(f"min(): {min(values4)}")

values5 = [1, 2, NINF, 3]
print(f"\nList: {values5}")
print(f"max(): {max(values5)}")
print(f"min(): {min(values5)}")

print("\n=== Testing transitivity ===")
# If a > b and b > c, then a > c should hold
a = AlwaysGreaterThan()
b = 10
c = 5
print(f"a = AlwaysGreaterThan(), b = 10, c = 5")
print(f"a > b: {a > b}")
print(f"b > c: {b > c}")
print(f"a > c: {a > c}")
print("Transitivity appears to hold")

print("\n=== Testing with identical instances ===")
# What about using the same instance multiple times?
inf1 = AlwaysGreaterThan()
values6 = [1, inf1, 2, inf1, 3]
print(f"List with same instance twice: {values6}")
try:
    sorted_values6 = sorted(values6)
    print(f"Sorted: {sorted_values6}")
except Exception as e:
    print(f"Error sorting: {e}")

print("\n=== Testing set operations (uniqueness) ===")
# Since they're equal, they should be deduplicated in a set
inf_a = AlwaysGreaterThan()
inf_b = AlwaysGreaterThan()
print(f"inf_a == inf_b: {inf_a == inf_b}")
print(f"inf_a is inf_b: {inf_a is inf_b}")
try:
    s = {inf_a, inf_b}
    print(f"Set with two AlwaysGreaterThan instances: {s}")
    print(f"Length of set: {len(s)}")
except Exception as e:
    print(f"Error creating set: {e}")