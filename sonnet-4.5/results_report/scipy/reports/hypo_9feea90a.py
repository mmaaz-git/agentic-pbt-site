#!/usr/bin/env python3
"""Property-based test demonstrating CombineKwargDefault hash instability."""

from hypothesis import given, strategies as st
import xarray
from xarray.util.deprecation_helpers import CombineKwargDefault

@given(
    name=st.text(min_size=1, max_size=20),
    old_val=st.text(min_size=1, max_size=10),
    new_val=st.text(min_size=1, max_size=10) | st.none()
)
def test_hash_stability(name, old_val, new_val):
    """Hash of an object should remain constant during its lifetime."""
    obj = CombineKwargDefault(name=name, old=old_val, new=new_val)

    xarray.set_options(use_new_combine_kwarg_defaults=False)
    hash1 = hash(obj)

    xarray.set_options(use_new_combine_kwarg_defaults=True)
    hash2 = hash(obj)

    assert hash1 == hash2, f"Hash changed from {hash1} to {hash2} when options changed"

# Run the test
if __name__ == "__main__":
    test_hash_stability()