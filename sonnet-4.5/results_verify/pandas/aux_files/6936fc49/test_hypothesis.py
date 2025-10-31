from hypothesis import given, strategies as st
import pytest
from pandas.api.indexers import FixedForwardWindowIndexer

@given(
    window_size=st.integers(min_value=1, max_value=100),
    num_values=st.integers(min_value=1, max_value=100),
)
def test_fixed_forward_indexer_step_zero_should_raise(window_size, num_values):
    indexer = FixedForwardWindowIndexer(window_size=window_size)

    with pytest.raises(ValueError, match="step must be"):
        indexer.get_window_bounds(num_values=num_values, step=0)

# Run the test
if __name__ == "__main__":
    # Test with the specific failing input
    print("Testing with window_size=1, num_values=1, step=0")
    try:
        test_fixed_forward_indexer_step_zero_should_raise(window_size=1, num_values=1)
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed with AssertionError: {e}")
    except ZeroDivisionError as e:
        print(f"Got ZeroDivisionError instead of ValueError: {e}")
    except Exception as e:
        print(f"Unexpected exception: {type(e).__name__}: {e}")

    # Let's also test a few more cases
    print("\nTesting with different values:")
    for window_size, num_values in [(5, 10), (10, 5), (50, 50)]:
        print(f"  window_size={window_size}, num_values={num_values}, step=0")
        try:
            test_fixed_forward_indexer_step_zero_should_raise(window_size, num_values)
            print("    Test passed!")
        except AssertionError as e:
            print(f"    Test failed - AssertionError")
        except ZeroDivisionError as e:
            print(f"    Got ZeroDivisionError instead of ValueError")