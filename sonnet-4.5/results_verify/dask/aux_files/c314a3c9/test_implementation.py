#!/usr/bin/env python3
"""Verify the exact behavior of memory_repr by reimplementing it"""

def memory_repr_copy(num):
    """Exact copy of the memory_repr function from dask/utils.py"""
    for x in ["bytes", "KB", "MB", "GB", "TB"]:
        if num < 1024.0:
            return f"{num:3.1f} {x}"
        num /= 1024.0
    # Function ends here - no explicit return statement!
    # Python implicitly returns None

# Test boundary cases
test_cases = [
    1024**4 - 1,  # Just under 1 TB
    1024**4,      # Exactly 1 TB
    1024**5 - 1,  # Just under 1024 TB
    1024**5,      # Exactly 1024 TB (1 PB)
]

print("Testing boundary cases:")
for val in test_cases:
    result = memory_repr_copy(val)
    if val < 1024**5:
        size_str = f"{val / (1024**4):.1f} TB"
    else:
        size_str = f"{val / (1024**5):.1f} PB"
    print(f"  {val:,} bytes ({size_str}) -> {result!r}")

# Verify what happens in the loop
print("\nTracing through the algorithm for 1024^5:")
num = 1024**5
print(f"  Initial: num = {num:,}")
for i, x in enumerate(["bytes", "KB", "MB", "GB", "TB"]):
    if num < 1024.0:
        print(f"  Step {i}: {num:.1f} < 1024, returning '{num:3.1f} {x}'")
        break
    print(f"  Step {i}: {num:.1f} >= 1024, dividing by 1024 (unit: {x})")
    num /= 1024.0
else:
    print(f"  Loop exhausted: num = {num:.1f}, no return statement, returning None")