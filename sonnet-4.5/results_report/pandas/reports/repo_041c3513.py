#!/usr/bin/env python3
"""
Minimal reproduction of the is_subperiod/is_superperiod inverse bug in pandas.tseries.frequencies
"""

from pandas.tseries.frequencies import is_subperiod, is_superperiod

# Test case 1: D (calendar day) and B (business day)
print("Test case 1: D and B")
print(f"is_superperiod('D', 'B') = {is_superperiod('D', 'B')}")
print(f"is_subperiod('B', 'D') = {is_subperiod('B', 'D')}")
print(f"  Expected: If superperiod returns True, subperiod should also return True")
print()

# Test case 2: D (calendar day) and C (custom business day)
print("Test case 2: D and C")
print(f"is_superperiod('D', 'C') = {is_superperiod('D', 'C')}")
print(f"is_subperiod('C', 'D') = {is_subperiod('C', 'D')}")
print(f"  Expected: If superperiod returns True, subperiod should also return True")
print()

# Test case 3: B (business day) and D (calendar day)
print("Test case 3: B and D")
print(f"is_superperiod('B', 'D') = {is_superperiod('B', 'D')}")
print(f"is_subperiod('D', 'B') = {is_subperiod('D', 'B')}")
print(f"  Expected: If superperiod returns True, subperiod should also return True")
print()

# Test case 4: B (business day) and C (custom business day)
print("Test case 4: B and C")
print(f"is_superperiod('B', 'C') = {is_superperiod('B', 'C')}")
print(f"is_subperiod('C', 'B') = {is_subperiod('C', 'B')}")
print(f"  Expected: If superperiod returns True, subperiod should also return True")
print()

# Test case 5: C (custom business day) and D (calendar day)
print("Test case 5: C and D")
print(f"is_superperiod('C', 'D') = {is_superperiod('C', 'D')}")
print(f"is_subperiod('D', 'C') = {is_subperiod('D', 'C')}")
print(f"  Expected: If superperiod returns True, subperiod should also return True")
print()

# Test case 6: C (custom business day) and B (business day)
print("Test case 6: C and B")
print(f"is_superperiod('C', 'B') = {is_superperiod('C', 'B')}")
print(f"is_subperiod('B', 'C') = {is_subperiod('B', 'C')}")
print(f"  Expected: If superperiod returns True, subperiod should also return True")