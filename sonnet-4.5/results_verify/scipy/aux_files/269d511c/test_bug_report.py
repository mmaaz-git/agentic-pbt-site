#!/usr/bin/env python3
"""Test the bug report about scipy.io.matlab.savemat variable name validation."""

import tempfile
import os
import numpy as np
from scipy.io.matlab import loadmat, savemat
import warnings

def test_hypothesis_test():
    """Run the hypothesis test from the bug report."""

    # Test with specific failing input
    print("Testing with var_name='0'...")
    arr = np.array([1.0, 2.0, 3.0])

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, 'test.mat')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            savemat(file_path, {'0': arr})

        loaded = loadmat(file_path)

        if '0' not in loaded:
            print("Test passed for var_name='0' - variable was NOT saved (as expected)")
        else:
            print(f"Test FAILED for var_name='0' - variable WAS saved (unexpected)")

    # Test with digit-starting names
    print("\nTesting various digit-starting names...")
    for name in ['0', '1', '2', '9', '5test', '9abc']:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, 'test.mat')

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                savemat(file_path, {name: arr})

            loaded = loadmat(file_path)

            if name in loaded:
                print(f"  '{name}': SAVED (found in file)")
            else:
                print(f"  '{name}': NOT SAVED (not found in file)")

def test_reproduction_code():
    """Run the reproduction code from the bug report."""
    print("\n=== Reproduction Code Test ===")

    arr = np.array([1.0, 2.0, 3.0])

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, 'test.mat')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            savemat(file_path, {'0': arr, '9abc': arr, 'abc': arr})
            print(f"Warnings issued: {len(w)}")
            if w:
                for warning in w:
                    print(f"  Warning: {warning.message}")

        loaded = loadmat(file_path)

        print("Variables saved:")
        for key in loaded.keys():
            if not key.startswith('__'):
                print(f"  {key}")

        print(f"\nVariable '0' in file: {'0' in loaded}")
        print(f"Variable '9abc' in file: {'9abc' in loaded}")
        print(f"Variable 'abc' in file: {'abc' in loaded}")

    print("\n=== Testing with underscore ===")
    # Test that underscore prefix DOES work as documented
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, 'test2.mat')

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            savemat(file_path, {'_private': arr, 'public': arr})
            print(f"Warnings issued for underscore test: {len(w)}")
            if w:
                for warning in w:
                    print(f"  Warning: {warning.message}")

        loaded = loadmat(file_path)
        print(f"Variable '_private' in file: {'_private' in loaded}")
        print(f"Variable 'public' in file: {'public' in loaded}")

if __name__ == "__main__":
    test_hypothesis_test()
    test_reproduction_code()