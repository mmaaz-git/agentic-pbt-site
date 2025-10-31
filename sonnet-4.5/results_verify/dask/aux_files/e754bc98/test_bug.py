from hypothesis import given, strategies as st
from dask.dataframe.io.io import sorted_division_locations

@given(
    seq=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=100),
    chunksize=st.integers(min_value=1, max_value=50)
)
def test_sorted_division_locations_basic_properties_chunksize(seq, chunksize):
    seq_sorted = sorted(seq)
    try:
        divisions, locations = sorted_division_locations(seq_sorted, chunksize=chunksize)

        assert locations[0] == 0
        assert locations[-1] == len(seq_sorted)
        assert divisions[0] == seq_sorted[0]
        assert divisions[-1] == seq_sorted[-1]
        print(f"Test passed for seq={seq_sorted[:5]}... (len={len(seq_sorted)}), chunksize={chunksize}")
    except TypeError as e:
        print(f"TypeError with seq={seq_sorted[:5]}... (len={len(seq_sorted)}), chunksize={chunksize}: {e}")
        raise

# Run a simple test
test_sorted_division_locations_basic_properties_chunksize()