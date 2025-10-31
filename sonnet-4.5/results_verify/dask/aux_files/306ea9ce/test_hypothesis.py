#!/usr/bin/env python3
"""Test the bug with proper Hypothesis setup"""

from dask.diagnostics.profile_visualize import unquote

# Manual test of the specific failing case
print("=== Manual test of empty dict task ===")
try:
    task = (dict, [])
    result = unquote(task)
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError confirmed: {e}")

# Test what we would expect
print("\n=== Testing expected behavior ===")
print("Empty dict created directly: ", dict([]))
print("This shows that dict([]) should return {}")

# Let's look at the actual function to understand what it's trying to do
print("\n=== Examining the unquote function behavior ===")

# Test different task formats
test_cases = [
    ("Empty dict task", (dict, [])),
    ("Dict with list pairs", (dict, [['a', 1], ['b', 2]])),
    ("Simple value", 42),
    ("String", "hello"),
    ("Tuple", (1, 2, 3)),
    ("List", [1, 2, 3]),
]

for description, task in test_cases:
    try:
        result = unquote(task)
        print(f"{description}: {task} -> {result}")
    except Exception as e:
        print(f"{description}: {task} -> ERROR: {type(e).__name__}: {e}")