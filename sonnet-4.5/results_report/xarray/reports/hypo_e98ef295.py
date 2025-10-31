from hypothesis import given, strategies as st
import numpy as np
from xarray.compat.array_api_compat import result_type, is_weak_scalar_type

@given(st.text(min_size=1, max_size=10))
def test_result_type_string_scalars_should_work(text):
    assert is_weak_scalar_type(text)
    result = result_type(text, xp=np)
    assert isinstance(result, np.dtype)

if __name__ == "__main__":
    test_result_type_string_scalars_should_work()