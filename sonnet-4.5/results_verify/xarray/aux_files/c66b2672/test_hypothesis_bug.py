from hypothesis import given, strategies as st
import numpy as np
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')
from xarray.compat.array_api_compat import result_type, is_weak_scalar_type

@given(st.text(min_size=1, max_size=10))
def test_result_type_string_scalars_should_work(text):
    print(f"Testing with text: {repr(text)}")
    assert is_weak_scalar_type(text)
    result = result_type(text, xp=np)
    assert isinstance(result, np.dtype)
    print(f"Result dtype: {result}")

# Run the test
test_result_type_string_scalars_should_work()