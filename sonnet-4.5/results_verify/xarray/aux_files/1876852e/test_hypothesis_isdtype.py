#!/usr/bin/env python3
import numpy as np
from hypothesis import given, strategies as st
import pytest
from xarray.compat import npcompat

@given(st.sampled_from([np.int32(1), np.float64(1.0), np.bool_(True)]))
def test_isdtype_scalar_handling_inconsistency(scalar):
    """Property test: isdtype should handle scalars consistently across numpy versions"""

    has_native_isdtype = hasattr(np, 'isdtype')

    if has_native_isdtype:
        # NumPy >= 2.0 behavior: should reject scalar values
        with pytest.raises(TypeError):
            npcompat.isdtype(scalar, 'numeric')
    else:
        # NumPy < 2.0 behavior: accepts scalar values (via fallback implementation)
        result = npcompat.isdtype(scalar, 'numeric')
        assert isinstance(result, bool)

# Run the test
if __name__ == "__main__":
    for scalar in [np.int32(5), np.float64(1.0), np.bool_(True)]:
        print(f"Testing scalar: {scalar} (type: {type(scalar)})")
        try:
            result = npcompat.isdtype(scalar, 'numeric')
            print(f"  Result: {result} (NumPy < 2.0 behavior)")
        except TypeError as e:
            print(f"  TypeError: {e} (NumPy >= 2.0 behavior)")
        print()