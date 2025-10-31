#!/usr/bin/env python3
"""
Minimal reproduction of the xarray._load_static_files cache mutation bug.
This demonstrates that the cached list can be corrupted by callers.
"""

from xarray.core.formatting_html import _load_static_files

# Get the original cached values
print("Getting the original cached values...")
original = _load_static_files()
print(f"Type of returned value: {type(original)}")
print(f"Number of elements: {len(original)}")
print(f"First element starts with: {original[0][:50]}...")
print(f"Second element starts with: {original[1][:50]}...")

# Store the original first element for comparison
original_first_element = original[0]

# Corrupt the cache by modifying the returned list
print("\nCorrupting the cache by modifying the list...")
original[0] = "CORRUPTED_CACHE_VALUE"

# Get the value again - it should be immutable but it's not
print("\nGetting the cached values again...")
second = _load_static_files()
print(f"First element is now: {second[0]}")
print(f"Are they the same object? {original is second}")

# Verify the corruption
print("\nVerifying the corruption...")
if second[0] == "CORRUPTED_CACHE_VALUE":
    print("BUG CONFIRMED: The cached value was corrupted!")
    print("The function returns a mutable list that shares state across calls.")
else:
    print("No bug: The cached value was not corrupted.")