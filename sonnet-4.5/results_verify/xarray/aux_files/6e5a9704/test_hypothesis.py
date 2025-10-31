import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from hypothesis import given, strategies as st, settings
from xarray.compat.array_api_compat import result_type


@given(st.text(max_size=10))
@settings(max_examples=20)
def test_result_type_with_str_scalar(value):
    print(f"Testing with string: {repr(value)}")
    result = result_type(value, xp=np)
    assert isinstance(result, np.dtype)


@given(st.binary(max_size=10))
@settings(max_examples=20)
def test_result_type_with_bytes_scalar(value):
    print(f"Testing with bytes: {repr(value)}")
    result = result_type(value, xp=np)
    assert isinstance(result, np.dtype)

if __name__ == "__main__":
    print("Running hypothesis tests for string scalars:")
    print("="*50)
    try:
        test_result_type_with_str_scalar()
        print("String test passed!")
    except Exception as e:
        print(f"String test failed: {e}")

    print("\nRunning hypothesis tests for bytes scalars:")
    print("="*50)
    try:
        test_result_type_with_bytes_scalar()
        print("Bytes test passed!")
    except Exception as e:
        print(f"Bytes test failed: {e}")