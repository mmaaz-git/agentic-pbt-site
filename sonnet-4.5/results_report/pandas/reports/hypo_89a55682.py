import pandas as pd
from pandas.core.arrays import IntervalArray
from hypothesis import given, strategies as st, assume, settings


@settings(max_examples=500)
@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=2, max_size=30),
    st.sampled_from(['left', 'right', 'both', 'neither'])
)
def test_intervalarray_unique_preserves_distinct_intervals(breaks, closed):
    breaks_sorted = sorted(set(breaks))
    assume(len(breaks_sorted) >= 2)

    arr = IntervalArray.from_breaks(breaks_sorted, closed=closed)
    combined = IntervalArray._concat_same_type([arr, arr])
    unique_arr = combined.unique()

    manual_unique = set()
    for interval in combined:
        manual_unique.add((interval.left, interval.right, interval.closed))

    assert len(unique_arr) == len(manual_unique), \
        f"unique() returned {len(unique_arr)} intervals but there are {len(manual_unique)} distinct intervals"

if __name__ == "__main__":
    test_intervalarray_unique_preserves_distinct_intervals()