#!/usr/bin/env python3
"""Run the property-based test from the bug report."""

import os
import shutil
import platformdirs
import pytest
from scipy import datasets

# Ensure cache directory doesn't exist
cache_dir = platformdirs.user_cache_dir("scipy-data")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

def test_clear_cache_validates_before_checking_directory():
    def fake_dataset():
        pass

    with pytest.raises(ValueError, match="Dataset method fake_dataset doesn't exist"):
        datasets.clear_cache(fake_dataset)

# Run the test
print("Running property-based test...")
try:
    test_clear_cache_validates_before_checking_directory()
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed as expected: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")