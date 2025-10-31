#!/usr/bin/env python3
"""Minimal reproduction of NBitBase deprecation warning bug"""

import warnings
import numpy.typing as npt

print("Testing NBitBase deprecation warning in numpy.typing")
print("=" * 60)

# Test 1: Direct access to NBitBase
print("\nTest 1: Direct access to NBitBase")
print("-" * 40)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = npt.NBitBase

    print(f"Accessed: npt.NBitBase")
    print(f"Result type: {type(result)}")
    print(f"Warnings caught: {len(w)}")

    if len(w) > 0:
        for warning in w:
            print(f"  Warning: {warning.category.__name__}: {warning.message}")
    else:
        print("  BUG: No deprecation warning emitted!")

# Test 2: Access via getattr
print("\nTest 2: Access via getattr()")
print("-" * 40)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    result = getattr(npt, "NBitBase")

    print(f"Accessed: getattr(npt, 'NBitBase')")
    print(f"Result type: {type(result)}")
    print(f"Warnings caught: {len(w)}")

    if len(w) > 0:
        for warning in w:
            print(f"  Warning: {warning.category.__name__}: {warning.message}")
    else:
        print("  BUG: No deprecation warning emitted!")

# Test 3: Check module internals
print("\nTest 3: Module internals")
print("-" * 40)
print(f"'NBitBase' in npt.__dict__: {'NBitBase' in npt.__dict__}")
print(f"'NBitBase' in npt.__all__: {'NBitBase' in npt.__all__}")

# Test 4: Expected behavior summary
print("\n" + "=" * 60)
print("SUMMARY:")
print("Expected: DeprecationWarning when accessing NBitBase")
print("Actual: No warning emitted")
print("\nRoot cause: NBitBase is imported directly into module namespace")
print("at line 160 of numpy/typing/__init__.py, bypassing the __getattr__")
print("hook (lines 173-184) that contains the deprecation warning.")