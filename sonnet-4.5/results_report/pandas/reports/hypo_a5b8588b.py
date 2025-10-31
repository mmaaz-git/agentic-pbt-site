from hypothesis import given, strategies as st, settings
import numpy as np
from pandas.core.indexers import length_of_indexer

@given(
    target=st.lists(st.integers(), min_size=1, max_size=50),
    slice_start=st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
    slice_stop=st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
    slice_step=st.one_of(st.none(), st.integers(min_value=1, max_value=10))
)
@settings(max_examples=100)
def test_length_of_indexer_slice_consistency(target, slice_start, slice_stop, slice_step):
    """Test that length_of_indexer returns the same length as actual numpy slicing."""
    target_array = np.array(target)
    indexer = slice(slice_start, slice_stop, slice_step)

    actual_length = len(target_array[indexer])
    predicted_length = length_of_indexer(indexer, target_array)

    assert actual_length == predicted_length, (
        f"Mismatch for {indexer} on array of length {len(target_array)}: "
        f"actual={actual_length}, predicted={predicted_length}"
    )

if __name__ == "__main__":
    # Run the property-based test
    print("Running property-based test for length_of_indexer...")
    print("Testing the property: length_of_indexer(indexer, target) == len(target[indexer])")
    print()

    try:
        test_length_of_indexer_slice_consistency()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed with assertion error: {e}")
    except Exception as e:
        print(f"Test failed with error: {e}")