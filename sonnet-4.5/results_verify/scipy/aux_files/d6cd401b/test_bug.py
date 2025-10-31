#!/usr/bin/env python3
"""Test to reproduce the reported bug."""

import os
import shutil
import platformdirs
from scipy import datasets

# First, ensure the cache directory doesn't exist
cache_dir = platformdirs.user_cache_dir("scipy-data")
print(f"Cache directory: {cache_dir}")

# Remove the cache directory if it exists
if os.path.exists(cache_dir):
    print(f"Removing existing cache directory: {cache_dir}")
    shutil.rmtree(cache_dir)

# Verify it doesn't exist
print(f"Cache directory exists: {os.path.exists(cache_dir)}")

# Now test the bug scenario
print("\n=== Testing with non-existent cache directory ===")

def fake_dataset():
    """A fake dataset function that doesn't exist in scipy."""
    pass

print("Calling clear_cache with invalid dataset method...")
try:
    datasets.clear_cache(fake_dataset)
    print("No exception was raised! Bug confirmed.")
except ValueError as e:
    print(f"ValueError raised as expected: {e}")
except Exception as e:
    print(f"Unexpected exception: {e}")

# Now create cache directory and test again
print("\n=== Creating cache directory and testing again ===")
os.makedirs(cache_dir, exist_ok=True)
print(f"Cache directory exists: {os.path.exists(cache_dir)}")

print("Calling clear_cache with invalid dataset method...")
try:
    datasets.clear_cache(fake_dataset)
    print("No exception was raised!")
except ValueError as e:
    print(f"ValueError raised as expected: {e}")
except Exception as e:
    print(f"Unexpected exception: {e}")

# Clean up
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)