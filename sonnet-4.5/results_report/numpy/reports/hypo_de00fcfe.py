#!/usr/bin/env python3
"""
Property-based test for numpy.base_repr padding behavior.
This test verifies that padding adds exactly the specified number of zeros.
"""
import numpy as np
from hypothesis import given, strategies as st, settings, example

@given(st.integers(min_value=0, max_value=10000),
       st.integers(min_value=2, max_value=36),
       st.integers(min_value=1, max_value=20))
@example(number=0, base=2, padding=1)  # The specific failing case
@settings(max_examples=100)
def test_base_repr_padding_adds_exact_zeros(number, base, padding):
    """Test that padding adds exactly N zeros to the left of the representation."""
    repr_with_padding = np.base_repr(number, base=base, padding=padding)
    repr_without_padding = np.base_repr(number, base=base, padding=0)
    expected_length = len(repr_without_padding) + padding
    assert len(repr_with_padding) == expected_length, \
        f"For number={number}, base={base}, padding={padding}: " \
        f"got length {len(repr_with_padding)}, expected {expected_length}. " \
        f"repr_with_padding='{repr_with_padding}', repr_without_padding='{repr_without_padding}'"

if __name__ == "__main__":
    # Run the test
    print("Running property-based test for numpy.base_repr padding...")
    print("=" * 60)
    try:
        test_base_repr_padding_adds_exact_zeros()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        print("\nThis test found that numpy.base_repr does not consistently")
        print("add the specified number of padding zeros for all inputs.")