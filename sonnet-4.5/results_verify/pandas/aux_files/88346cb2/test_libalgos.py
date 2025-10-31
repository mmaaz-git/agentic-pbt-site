#!/usr/bin/env python3
"""Test libalgos.Infinity behavior"""

from pandas._libs import algos as libalgos

Inf = libalgos.Infinity()
print("Testing libalgos.Infinity():")
print(f"Inf == Inf: {Inf == Inf}")
print(f"Inf > Inf: {Inf > Inf}")
print(f"Inf < Inf: {Inf < Inf}")
print(f"Inf >= Inf: {Inf >= Inf}")
print(f"Inf <= Inf: {Inf <= Inf}")

# Check if the test from pandas passes
try:
    assert Inf >= Inf and Inf == Inf
    print("✓ assert Inf >= Inf and Inf == Inf PASSED")
except AssertionError:
    print("✗ assert Inf >= Inf and Inf == Inf FAILED")

try:
    assert not Inf < Inf and not Inf > Inf
    print("✓ assert not Inf < Inf and not Inf > Inf PASSED")
except AssertionError:
    print("✗ assert not Inf < Inf and not Inf > Inf FAILED")