#!/usr/bin/env python3
"""Test script to reproduce the scipy.constants.precision bug"""

from hypothesis import given, strategies as st
from scipy.constants import precision, find, physical_constants

# First, let's run the hypothesis test
print("Running hypothesis test...")
@given(st.sampled_from(find()))
def test_precision_is_always_nonnegative(key):
    prec = precision(key)
    assert prec >= 0, f"precision('{key}') returned {prec}, which is negative"

try:
    test_precision_is_always_nonnegative()
    print("Hypothesis test passed!")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")

print("\n" + "="*60)
print("Reproducing the specific bug case:")
print("="*60)

# Now let's reproduce the specific example
key = 'neutron to shielded proton mag. mom. ratio'
prec = precision(key)
print(f"precision('{key}') = {prec}")

value, unit, uncertainty = physical_constants[key]
print(f"Value: {value}")
print(f"Uncertainty: {uncertainty}")
print(f"Calculated precision: {uncertainty / value}")
print(f"Expected (non-negative): {abs(uncertainty / value)}")

print("\n" + "="*60)
print("Checking for other negative precision values:")
print("="*60)

# Let's find all constants with negative precision
negative_precision_constants = []
for key in find():
    prec = precision(key)
    if prec < 0:
        negative_precision_constants.append((key, prec, physical_constants[key][0]))

print(f"Found {len(negative_precision_constants)} constants with negative precision:")
for i, (k, p, v) in enumerate(negative_precision_constants[:5], 1):
    print(f"{i}. '{k}': precision={p:.3e}, value={v:.3e}")
if len(negative_precision_constants) > 5:
    print(f"... and {len(negative_precision_constants) - 5} more")