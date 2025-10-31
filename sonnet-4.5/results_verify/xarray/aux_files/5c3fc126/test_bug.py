#!/usr/bin/env python3
"""Test the purported bug in xarray's _load_static_files function"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.core import formatting_html as fh

print("Testing _load_static_files cache mutability bug...")
print("=" * 60)

# First call - get the original values
first_call = fh._load_static_files()
print(f"Type of return value: {type(first_call)}")
print(f"Number of elements: {len(first_call)}")
print(f"First element (first 50 chars): {first_call[0][:50]}")
print(f"Object ID of first call: {id(first_call)}")

# Save the original value for comparison
original_first_element = first_call[0]
print(f"\nOriginal first element saved (first 50 chars): {original_first_element[:50]}")

# Second call - should return the same cached object
second_call = fh._load_static_files()
print(f"\nObject ID of second call: {id(second_call)}")
print(f"Are first and second calls the same object? {first_call is second_call}")

# Now mutate the second call's result
print("\n" + "=" * 60)
print("Mutating the second call's first element to 'MUTATED'...")
second_call[0] = "MUTATED"

# Third call - check if mutation affected the cache
third_call = fh._load_static_files()
print(f"\nObject ID of third call: {id(third_call)}")
print(f"Third call's first element: {third_call[0]}")

# Check if all calls now have the mutated value
print("\n" + "=" * 60)
print("Checking all references:")
print(f"First call's first element: {first_call[0]}")
print(f"Second call's first element: {second_call[0]}")
print(f"Third call's first element: {third_call[0]}")

# Verify the bug
print("\n" + "=" * 60)
if third_call[0] == "MUTATED":
    print("BUG CONFIRMED: Cache was mutated by external modification!")
    print("The cached list is mutable and shared across all calls.")
else:
    print("BUG NOT REPRODUCED: Cache was not affected by modification.")

# Also test the property-based test from the bug report
print("\n" + "=" * 60)
print("Running the property-based test from bug report...")

def test_load_static_files_cache_immutability():
    first_call = fh._load_static_files()
    original_value = first_call[0]

    second_call = fh._load_static_files()
    second_call[0] = "MUTATED_IN_TEST"

    third_call = fh._load_static_files()

    assert third_call[0] == original_value, "Cache should not be mutated by callers"

try:
    # Clear the cache first
    fh._load_static_files.cache_clear()
    test_load_static_files_cache_immutability()
    print("Property-based test PASSED (bug not present)")
except AssertionError as e:
    print(f"Property-based test FAILED: {e}")
    print("This confirms the bug - cache can be mutated by callers")