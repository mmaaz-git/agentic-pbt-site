#!/usr/bin/env python3
"""Test the reported bug in scipy.datasets.face gray parameter"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages')

from scipy import datasets
import numpy as np

print("Testing face() with different truthy values:")
print()

test_values = [
    ("True", True),
    ("False", False),
    ("1", 1),
    ("0", 0),
    ("np.bool_(True)", np.bool_(True)),
    ("np.bool_(False)", np.bool_(False)),
]

for name, value in test_values:
    try:
        result = datasets.face(gray=value)
        print(f"gray={name:20s} -> shape {result.shape}")
    except Exception as e:
        print(f"gray={name:20s} -> ERROR: {e}")

print("\n" + "="*60)
print("Property-based test from bug report:")
print("="*60 + "\n")

from hypothesis import given, strategies as st

@given(st.sampled_from([True, 1, "true", np.bool_(True)]))
def test_face_gray_accepts_truthy_values(gray_value):
    """Property: face(gray=X) should accept any truthy value, not just True."""
    from scipy import datasets
    import numpy as np

    try:
        result = datasets.face(gray=gray_value)

        if gray_value:
            expected_shape = (768, 1024)
        else:
            expected_shape = (768, 1024, 3)

        assert result.shape == expected_shape, \
            f"gray={gray_value!r} produced shape {result.shape}, expected {expected_shape}"
        print(f"✓ gray={gray_value!r} -> shape {result.shape}")
    except Exception as e:
        print(f"✗ Failed with gray={gray_value!r}: {e}")
        raise

# Run the property test
try:
    print("Running property-based test...")
    test_face_gray_accepts_truthy_values()
except Exception as e:
    print(f"Property test failed: {e}")