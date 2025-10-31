"""Test the hypothesis test from the bug report"""

from hypothesis import given, strategies as st, settings
from dask.dataframe.io.io import sorted_division_locations

@given(
    st.lists(st.integers(), min_size=1),
    st.integers(min_value=1, max_value=100)
)
@settings(max_examples=10)
def test_sorted_division_locations_accepts_lists(seq, chunksize):
    print(f"Testing with seq={seq[:5]}... (len={len(seq)}), chunksize={chunksize}")
    divisions, locations = sorted_division_locations(seq, chunksize=chunksize)
    assert locations[0] == 0
    assert locations[-1] == len(seq)

# Run the test
try:
    test_sorted_division_locations_accepts_lists()
    print("All tests passed!")
except Exception as e:
    print(f"Test failed: {e}")