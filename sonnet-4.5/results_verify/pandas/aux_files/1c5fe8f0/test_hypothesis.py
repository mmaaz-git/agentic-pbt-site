from hypothesis import given, strategies as st, assume
from hypothesis.extra import numpy as npst
import numpy as np
from pandas.api import extensions

@given(
    npst.arrays(dtype=np.int64, shape=npst.array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=20)),
    st.lists(st.integers(), min_size=1, max_size=10)
)
def test_take_out_of_bounds_raises(arr, indices):
    assume(len(arr) > 0)
    assume(any(idx >= len(arr) or idx < -len(arr) for idx in indices))

    try:
        extensions.take(arr, indices)
        assert False, "Expected IndexError"
    except IndexError:
        pass

# Run the test with the specific failing example manually
print("Testing with the specific failing input from the bug report...")
arr = np.array([0])
indices = [9_223_372_036_854_775_808]

print(f"arr = {arr}")
print(f"indices = {indices}")

# Manually test the logic instead of calling the decorated function
try:
    extensions.take(arr, indices)
    print("Test failed - Expected IndexError but no exception was raised")
except IndexError:
    print("Test passed - Got expected IndexError")
except OverflowError as e:
    print(f"Test failed - Got OverflowError instead of IndexError: {e}")
except Exception as e:
    print(f"Test failed with unexpected exception {type(e).__name__}: {e}")

# Also run the Hypothesis test automatically for a few examples
print("\nRunning property-based test with Hypothesis...")
try:
    test_take_out_of_bounds_raises()
    print("Property-based test completed successfully")
except Exception as e:
    print(f"Property-based test failed: {type(e).__name__}: {e}")