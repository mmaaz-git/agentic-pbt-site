#!/usr/bin/env python3
"""Property-based test demonstrating scipy.datasets.face() bug with truthy values."""

from hypothesis import given, strategies as st, settings
import scipy.datasets

@given(st.one_of(
    st.integers(min_value=1, max_value=10),
    st.text(min_size=1, max_size=5).filter(bool),
    st.lists(st.integers(), min_size=1, max_size=3)
))
@settings(max_examples=10)
def test_face_gray_truthy_values(val):
    """Test that truthy values for gray parameter should produce grayscale images."""
    result = scipy.datasets.face(gray=val)

    # All truthy values should trigger grayscale conversion
    if val:
        assert result.ndim == 2, \
            f"Truthy value {val!r} (type: {type(val).__name__}) should trigger grayscale conversion but returned shape {result.shape}"
        assert result.shape == (768, 1024), \
            f"Expected grayscale shape (768, 1024), got {result.shape}"
    else:
        # Falsy values should return color
        assert result.ndim == 3, \
            f"Falsy value {val!r} should return color image but returned shape {result.shape}"
        assert result.shape == (768, 1024, 3), \
            f"Expected color shape (768, 1024, 3), got {result.shape}"

if __name__ == "__main__":
    test_face_gray_truthy_values()