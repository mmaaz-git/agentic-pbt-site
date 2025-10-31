from hypothesis import given, strategies as st, assume
import numpy as np
from xarray.indexes import RangeIndex

@given(
    start=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    stop=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    num=st.integers(min_value=2, max_value=1000),
)
def test_linspace_endpoint_true_last_value_equals_stop(start, stop, num):
    assume(abs(stop - start) > 1e-6)

    index = RangeIndex.linspace(start, stop, num, endpoint=True, dim="x")
    values = index.transform.forward({"x": np.arange(num)})["x"]

    assert values[-1] == stop, f"Last value {values[-1]} != stop {stop}"

# Run the specific failing case
print("Testing the specific failing case from the bug report:")
def test_specific_case():
    start, stop, num = 817040.0, 0.0, 18
    index = RangeIndex.linspace(start, stop, num, endpoint=True, dim="x")
    values = index.transform.forward({"x": np.arange(num)})["x"]
    assert values[-1] == stop, f"Last value {values[-1]} != stop {stop}"

try:
    test_specific_case()
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed: {e}")

# Try running hypothesis testing
print("\nRunning hypothesis testing (will run several examples):")
from hypothesis import settings

# Run a few test cases
try:
    with settings(max_examples=10, verbosity=2):
        test_linspace_endpoint_true_last_value_equals_stop()
    print("All hypothesis tests passed!")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")