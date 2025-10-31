#!/usr/bin/env python3
"""Test script to reproduce the memory_repr bug"""

import dask.dataframe.dask_expr as de

# Test 1: Check the function works for normal values
print("Testing normal values:")
for power in range(5):
    value = 1024 ** power
    result = de.memory_repr(value)
    print(f"  1024^{power} = {value:,} bytes -> {result!r}")

print("\nTesting the reported bug:")
# Test 2: The specific bug case - 1 PB (1024^5)
result = de.memory_repr(1024**5)
print(f"  1024^5 (1 PB) = {1024**5:,} bytes -> {result!r}")

# Test 3: Check if it returns None
if result is None:
    print("  ✗ Bug confirmed: memory_repr returns None for values > 1024 TB")
else:
    print(f"  ✓ No bug: memory_repr returned {result!r}")

# Test 4: Try even larger values
print("\nTesting even larger values:")
for power in [6, 7, 10]:
    value = 1024 ** power
    result = de.memory_repr(value)
    print(f"  1024^{power} = {value:.2e} bytes -> {result!r}")

# Test 5: Run the hypothesis test manually with the failing input
print("\nTesting the specific failing input from hypothesis:")
failing_input = 1125899906842624.0  # From the bug report
result = de.memory_repr(failing_input)
print(f"  {failing_input:.0f} bytes -> {result!r}")
assert result is not None, "memory_repr should never return None"