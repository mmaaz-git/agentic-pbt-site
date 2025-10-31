#!/usr/bin/env python3
"""
Property-based test for scipy.io.matlab digit-prefixed key bug
"""
from io import BytesIO
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.io.matlab import loadmat, savemat
import warnings


@settings(max_examples=50)
@given(st.from_regex(r'^[0-9][a-zA-Z0-9_]*$', fullmatch=True))
def test_digit_key_not_saved(key):
    """
    Test that keys starting with digits are not saved and trigger a warning.

    According to savemat documentation:
    "Note that if this dict has a key starting with `_` or a sub-dict has a key
    starting with `_` or a digit, these key's items will not be saved in the mat
    file and `MatWriteWarning` will be issued."
    """
    bio = BytesIO()
    data = {key: np.array([1, 2, 3])}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        savemat(bio, data)

        # Check if warning was issued
        if len(w) > 0:
            assert any("MatWriteWarning" in str(warn.category) for warn in w), \
                f"Expected MatWriteWarning but got: {[str(warn.category) for warn in w]}"

    bio.seek(0)
    loaded = loadmat(bio)

    # Check that key was not saved
    assert key not in loaded, f"Key '{key}' should not have been saved but was found in loaded data"


if __name__ == "__main__":
    # Run the test
    test_digit_key_not_saved()