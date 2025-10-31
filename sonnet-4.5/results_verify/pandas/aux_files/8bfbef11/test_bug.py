import pandas as pd
import numpy as np
from hypothesis import given, strategies as st
import traceback


@st.composite
def int_arrow_arrays(draw, min_size=0, max_size=30):
    data = draw(st.lists(
        st.one_of(st.integers(min_value=-1000, max_value=1000), st.none()),
        min_size=min_size,
        max_size=max_size
    ))
    return pd.array(data, dtype='int64[pyarrow]')


@given(arr=int_arrow_arrays(min_size=1, max_size=20))
def test_take_empty_indices(arr):
    """Taking with empty indices should return empty array."""
    result = arr.take([])
    assert len(result) == 0
    assert result.dtype == arr.dtype


print("=== Testing ArrowExtensionArray.take([]) bug ===")
print("\n1. Reproducing with provided code:")
try:
    arr = pd.array([1, 2, 3], dtype='int64[pyarrow]')
    print(f"Created array: {arr}")
    print(f"Array type: {type(arr)}")
    result = arr.take([])
    print(f"Result of take([]): {result}")
    print(f"Result length: {len(result)}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n2. Testing with regular pandas array:")
try:
    regular_arr = pd.array([1, 2, 3], dtype='int64')
    print(f"Created regular array: {regular_arr}")
    print(f"Array type: {type(regular_arr)}")
    result = regular_arr.take([])
    print(f"Result of take([]): {result}")
    print(f"Result length: {len(result)}")
    print(f"Result dtype: {result.dtype}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")

print("\n3. Testing what numpy creates from empty list:")
empty_indices = []
indices_array = np.asanyarray(empty_indices)
print(f"np.asanyarray([]) produces: {indices_array}")
print(f"dtype: {indices_array.dtype}")
print(f"shape: {indices_array.shape}")

print("\n4. Testing with explicit integer dtype:")
indices_array_int = np.asanyarray(empty_indices, dtype=np.intp)
print(f"np.asanyarray([], dtype=np.intp) produces: {indices_array_int}")
print(f"dtype: {indices_array_int.dtype}")

print("\n5. Running Hypothesis test:")
try:
    test_take_empty_indices()
    print("Hypothesis test completed")
except Exception as e:
    print(f"Hypothesis test failed with: {e}")
    traceback.print_exc()