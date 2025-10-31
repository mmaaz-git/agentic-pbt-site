#!/usr/bin/env python3
"""Test script to reproduce the InfinityType comparison bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

# Test 1: Run the property-based test
print("=" * 60)
print("Running property-based test...")
print("=" * 60)

from hypothesis import given, strategies as st
import pandas.util.version as version_module

@given(st.sampled_from([
    version_module.Infinity,
    version_module.NegativeInfinity
]))
def test_comparison_reflexivity(x):
    """Test that comparison operators are reflexive."""
    if x == x:
        assert x <= x, f"{x} should be <= itself when it equals itself"
        assert x >= x, f"{x} should be >= itself when it equals itself"

try:
    test_comparison_reflexivity()
    print("Property test PASSED")
except AssertionError as e:
    print(f"Property test FAILED: {e}")
except Exception as e:
    print(f"Unexpected error in property test: {e}")

# Test 2: Reproduce the specific bug manually
print("\n" + "=" * 60)
print("Running manual reproduction test...")
print("=" * 60)

inf = version_module.Infinity
print(f"Infinity == Infinity: {inf == inf}")
print(f"Infinity <= Infinity: {inf <= inf}")
print(f"Infinity >= Infinity: {inf >= inf}")

neginf = version_module.NegativeInfinity
print(f"\nNegativeInfinity == NegativeInfinity: {neginf == neginf}")
print(f"NegativeInfinity <= NegativeInfinity: {neginf <= neginf}")
print(f"NegativeInfinity >= NegativeInfinity: {neginf >= neginf}")

# Test 3: Check the fundamental property
print("\n" + "=" * 60)
print("Checking comparison consistency...")
print("=" * 60)

# The mathematical property: if a == b, then a <= b and a >= b must be true
inf_eq_inf = inf == inf
inf_le_inf = inf <= inf
inf_ge_inf = inf >= inf

neginf_eq_neginf = neginf == neginf
neginf_le_neginf = neginf <= neginf
neginf_ge_neginf = neginf >= neginf

print(f"\nFor Infinity:")
print(f"  Infinity == Infinity: {inf_eq_inf}")
print(f"  Infinity <= Infinity: {inf_le_inf}")
print(f"  Infinity >= Infinity: {inf_ge_inf}")
print(f"  Consistency check: {'PASS' if (not inf_eq_inf or (inf_le_inf and inf_ge_inf)) else 'FAIL'}")

print(f"\nFor NegativeInfinity:")
print(f"  NegativeInfinity == NegativeInfinity: {neginf_eq_neginf}")
print(f"  NegativeInfinity <= NegativeInfinity: {neginf_le_neginf}")
print(f"  NegativeInfinity >= NegativeInfinity: {neginf_ge_neginf}")
print(f"  Consistency check: {'PASS' if (not neginf_eq_neginf or (neginf_le_neginf and neginf_ge_neginf)) else 'FAIL'}")

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)

if inf_eq_inf and not inf_le_inf:
    print("BUG CONFIRMED: Infinity == Infinity is True but Infinity <= Infinity is False")
else:
    print("Infinity comparison is consistent")

if inf_eq_inf and not inf_ge_inf:
    print("BUG CONFIRMED: Infinity == Infinity is True but Infinity >= Infinity is False")
else:
    print("Infinity >= comparison is consistent")

if neginf_eq_neginf and not neginf_le_neginf:
    print("BUG CONFIRMED: NegativeInfinity == NegativeInfinity is True but NegativeInfinity <= NegativeInfinity is False")
else:
    print("NegativeInfinity <= comparison is consistent")

if neginf_eq_neginf and not neginf_ge_neginf:
    print("BUG CONFIRMED: NegativeInfinity == NegativeInfinity is True but NegativeInfinity >= NegativeInfinity is False")
else:
    print("NegativeInfinity >= comparison is consistent")