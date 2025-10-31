#!/usr/bin/env python3

from hypothesis import given, strategies as st, settings
import pandas.core.indexers as indexers

@given(
    start=st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
    stop=st.one_of(st.none(), st.integers(min_value=-50, max_value=50)),
    step=st.integers(min_value=-20, max_value=20).filter(lambda x: x != 0),
    n=st.integers(min_value=1, max_value=50)
)
@settings(max_examples=100)
def test_length_of_indexer_slice_comprehensive(start, stop, step, n):
    slc = slice(start, stop, step)
    target = list(range(n))

    computed_length = indexers.length_of_indexer(slc, target)
    actual_length = len(target[slc])

    assert computed_length == actual_length, \
        f"slice({start}, {stop}, {step}) on list of length {n}: " \
        f"length_of_indexer returned {computed_length}, actual length is {actual_length}"

# Run the test
try:
    test_length_of_indexer_slice_comprehensive()
    print("All hypothesis tests passed!")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")

# Also test the specific case mentioned in the bug report
print("\nSpecific test case from bug report:")
slc = slice(None, None, -1)
target = list(range(5))
computed = indexers.length_of_indexer(slc, target)
actual = len(target[slc])
print(f"slice(None, None, -1) with n=5:")
print(f"  Computed: {computed}")
print(f"  Actual: {actual}")
print(f"  Match: {computed == actual}")