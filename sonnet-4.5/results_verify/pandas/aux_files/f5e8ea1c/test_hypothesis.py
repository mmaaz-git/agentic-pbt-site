import numpy as np
from hypothesis import given, strategies as st
from pandas.core.indexers import length_of_indexer


@given(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=-100, max_value=100).filter(lambda x: x != 0),
    st.integers(min_value=-100, max_value=100),
    st.integers(min_value=-100, max_value=100),
)
def test_length_of_indexer_slice_matches_actual_length(target_len, step, start, stop):
    target = list(range(target_len))
    slc = slice(start, stop, step)

    calculated_length = length_of_indexer(slc, target)
    actual_slice = target[slc]
    actual_length = len(actual_slice)

    assert calculated_length == actual_length, \
        f"length_of_indexer({slc}, target={target_len}) = {calculated_length}, but len(target[{slc}]) = {actual_length}"

# Run the test
if __name__ == "__main__":
    # Test with the specific failing input from the report
    target_len = 1
    step = 1
    start = 1
    stop = 0

    print(f"Testing with: target_len={target_len}, step={step}, start={start}, stop={stop}")

    # Manually test the failing case
    target = list(range(target_len))
    slc = slice(start, stop, step)

    calculated_length = length_of_indexer(slc, target)
    actual_slice = target[slc]
    actual_length = len(actual_slice)

    print(f"Target: {target}")
    print(f"Slice: {slc}")
    print(f"Actual slice result: {actual_slice}")
    print(f"Calculated length: {calculated_length}")
    print(f"Actual length: {actual_length}")

    if calculated_length != actual_length:
        print(f"ERROR: length_of_indexer({slc}, target={target_len}) = {calculated_length}, but len(target[{slc}]) = {actual_length}")