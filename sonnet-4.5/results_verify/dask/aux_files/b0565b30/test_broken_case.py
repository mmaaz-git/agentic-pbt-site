#!/usr/bin/env python3
"""Find a case where the bug actually causes incorrect results"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.bag.core import accumulate_part
from operator import add
from tlz import first, second

print("Looking for cases where the bug causes wrong results")
print("=" * 60)

# The bug happens when is_first=True
# The return statement is: return res, res[-1] if res else [], initial

# Case 1: Empty sequence, but res becomes [initial] due to accumulate
print("\nCase 1: Empty sequence with initial value")
result = accumulate_part(add, [], 10, is_first=True)
print(f"accumulate_part(add, [], 10, is_first=True) = {result}")
print(f"Expected: ([10], 10)")
print(f"Actual: {result}")
if result == ([10], 10):
    print("✓ Correct despite the bug syntax")
else:
    print(f"✗ Wrong! 3-tuple: {len(result) == 3}")

# The issue is when res is non-empty, it returns (res, res[-1], initial)
# But we want (res, res[-1])
# This means the initial value gets included as a third element

# When does this break things?
print("\n" + "=" * 60)
print("The actual problem:")
print("When res is non-empty, line 1740 returns:")
print("  (res, res[-1], initial) - a 3-tuple")
print("But it should return:")
print("  (res, res[-1]) - a 2-tuple")
print()
print("The usage in accumulate() extracts elements with first() and second()")
print("first(3-tuple) gets element 0 (correct)")
print("second(3-tuple) gets element 1 (happens to be correct!)")
print()
print("So why doesn't it break? Because for a 3-tuple (a, b, c):")
print("  first() returns a (correct - the accumulated results)")
print("  second() returns b (correct - the last value res[-1])")
print("  The third element c (initial) is ignored!")

print("\n" + "=" * 60)
print("Wait, let me check if the bug report's example is wrong...")

# The bug report claims empty list gives ([], [], 10)
# But that's wrong - empty list with initial gives res=[10]

from itertools import accumulate
res = list(accumulate([], initial=10))
print(f"\nlist(accumulate([], initial=10)) = {res}")

# So when seq=[], res=[10] (not empty!)
# Therefore res[-1] if res else [] evaluates to res[-1] = 10

# Let's trace it:
seq = []
initial = 10
res = [10]  # from accumulate

# The problematic line:
# return res, res[-1] if res else [], initial

# Python parses this as:
# return (res, (res[-1] if res else []), initial)
# = ([10], 10, 10)

print(f"\nWith seq=[], initial=10:")
print(f"  res = {res}")
print(f"  res is truthy, so res[-1] = {res[-1]}")
print(f"  Returns: ({res}, {res[-1]}, {initial})")
print(f"  = ([10], 10, 10)")

print("\n" + "=" * 60)
print("CONCLUSION:")
print("The bug EXISTS - the function returns a 3-tuple instead of 2-tuple")
print("BUT it doesn't cause incorrect results because:")
print("1. first() and second() extract the right elements")
print("2. The third element is simply ignored")
print()
print("However, this is still a bug because:")
print("1. It's syntactically incorrect - missing parentheses")
print("2. Returns wrong tuple size (3 instead of 2)")
print("3. Could break if code expects exactly 2 elements")
print("4. Wastes memory with unnecessary third element")