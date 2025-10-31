#!/usr/bin/env python3
"""Test to verify the attribute lookup behavior"""

import warnings
import numpy.typing as npt

# Test 1: Check if NBitBase exists in module dict
print("Test 1: NBitBase in module dict")
print(f"'NBitBase' in dir(npt): {'NBitBase' in dir(npt)}")
print(f"hasattr(npt, 'NBitBase'): {hasattr(npt, 'NBitBase')}")
print(f"'NBitBase' in npt.__dict__: {'NBitBase' in npt.__dict__}")

# Test 2: Direct access using __getattr__
print("\nTest 2: Forcing __getattr__ call")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    # Try to call __getattr__ directly
    try:
        result = npt.__getattr__("NBitBase")
        print(f"__getattr__ returned: {result}")
        print(f"Warnings captured via __getattr__: {len(w)}")
        if w:
            print(f"Warning message: {w[0].message}")
    except AttributeError as e:
        print(f"AttributeError: {e}")

# Test 3: Remove from dict and try again
print("\nTest 3: Simulating the fix by removing from __dict__")
# Save the original
original_nbitbase = npt.__dict__.get('NBitBase')
if 'NBitBase' in npt.__dict__:
    # Temporarily remove it
    del npt.__dict__['NBitBase']

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            # Now it should go through __getattr__
            obj = npt.NBitBase
            print(f"Access after removal - Warnings captured: {len(w)}")
            if w:
                print(f"Warning message: {w[0].message}")
        except AttributeError as e:
            print(f"AttributeError: {e}")

    # Restore it
    if original_nbitbase is not None:
        npt.__dict__['NBitBase'] = original_nbitbase