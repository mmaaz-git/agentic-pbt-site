#!/usr/bin/env python3
"""Test script to reproduce the NBitBase deprecation warning bug."""

import warnings
import numpy.typing as npt
from hypothesis import given, strategies as st


# First, let's run the hypothesis test
@given(st.just("NBitBase"))
def test_nbitbase_emits_deprecation_warning(attr_name):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = getattr(npt, attr_name)

        assert len(w) >= 1, f"Expected deprecation warning for {attr_name}"
        assert any("deprecated" in str(w_item.message).lower() for w_item in w)


print("Running Hypothesis test:")
try:
    test_nbitbase_emits_deprecation_warning()
    print("Hypothesis test PASSED (no assertion error)")
except AssertionError as e:
    print(f"Hypothesis test FAILED: {e}")


# Now run the reproducer
print("\n" + "="*50)
print("Running reproducer code:")
print("="*50 + "\n")

print("Accessing npt.NBitBase (should emit deprecation warning but doesn't):")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.NBitBase
    print(f"Warnings raised: {len(w)}")
    if w:
        print(f"Warning message: {w[0].message}")

print("\nCalling __getattr__ directly (emits warning as intended):")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    _ = npt.__getattr__("NBitBase")
    print(f"Warnings raised: {len(w)}")
    if w:
        print(f"Warning message: {w[0].message}")

print("\nVerifying that NBitBase is indeed in module __dict__:")
print(f"'NBitBase' in npt.__dict__: {'NBitBase' in npt.__dict__}")
print(f"'NBitBase' in dir(npt): {'NBitBase' in dir(npt)}")