#!/usr/bin/env python3
"""Test to demonstrate the identity check vs truthiness check issue"""

import numpy as np

# Test values that are truthy
test_values = [
    ("True", True),
    ("1", 1),
    ("np.bool_(True)", np.bool_(True)),
    ('"true"', "true"),
    ("1.0", 1.0),
    ("[1]", [1]),
]

print("Identity check (is True) vs Truthiness check (if x):")
print("="*60)

for name, value in test_values:
    # Identity check
    identity_result = value is True

    # Truthiness check
    truthiness_result = bool(value)

    print(f"Value: {name:20s}")
    print(f"  'is True': {identity_result}")
    print(f"  'bool(x)': {truthiness_result}")
    print(f"  Mismatch: {'YES' if identity_result != truthiness_result else 'NO'}")
    print()

print("\nCurrent scipy.datasets.face behavior:")
print("Uses 'if gray is True:' which only accepts the literal Python True")
print("This rejects truthy values like 1, np.bool_(True), etc.")