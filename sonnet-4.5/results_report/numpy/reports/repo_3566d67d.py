#!/usr/bin/env python3
"""Demonstration of NBitBase deprecation warning not firing."""

import warnings
import numpy.typing as npt

print("Test 1: Accessing npt.NBitBase directly (normal usage)")
print("-" * 60)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.NBitBase
    print(f"Number of warnings captured: {len(w)}")
    if len(w) > 0:
        for warning in w:
            print(f"Warning: {warning.category.__name__}: {warning.message}")
    else:
        print("NO WARNINGS ISSUED - This is the bug!")

print("\n" + "=" * 60 + "\n")

print("Test 2: Accessing NBitBase via __getattr__ (forced)")
print("-" * 60)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.__getattr__('NBitBase')
    print(f"Number of warnings captured: {len(w)}")
    if len(w) > 0:
        for warning in w:
            print(f"Warning: {warning.category.__name__}: {warning.message}")
    else:
        print("No warnings issued")

print("\n" + "=" * 60 + "\n")

print("Verification: Why does this happen?")
print("-" * 60)
print(f"NBitBase in npt.__all__: {'NBitBase' in npt.__all__}")
print(f"NBitBase in module globals: {'NBitBase' in dir(npt)}")
print(f"NBitBase is directly accessible: {hasattr(npt, 'NBitBase')}")
print("\nExplanation: Since NBitBase is directly imported into the module")
print("namespace, Python finds it immediately without calling __getattr__.")
print("The deprecation warning code in __getattr__ is therefore unreachable.")