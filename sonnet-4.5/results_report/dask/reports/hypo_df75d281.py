"""Property-based test for sorted_division_locations with plain Python lists."""
from hypothesis import given, strategies as st, settings
from hypothesis import reproduce_failure
from dask.dataframe.io.io import sorted_division_locations

@given(
    seq=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=100),
    npartitions=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=10)
def test_sorted_division_locations_accepts_lists(seq, npartitions):
    """Test that sorted_division_locations accepts plain Python lists as documented."""
    seq_sorted = sorted(seq)
    try:
        divisions, locations = sorted_division_locations(seq_sorted, npartitions=npartitions)
        assert len(divisions) == len(locations)
        print(f"✓ Passed with seq={seq_sorted[:5]}{'...' if len(seq_sorted) > 5 else ''}, npartitions={npartitions}")
    except TypeError as e:
        if "No dispatch for <class 'list'>" in str(e):
            print(f"✗ Failed with seq={seq_sorted[:5]}{'...' if len(seq_sorted) > 5 else ''}, npartitions={npartitions}")
            print(f"  Error: {e}")
            raise
        else:
            raise

# Run the test
if __name__ == "__main__":
    print("Running property-based test for sorted_division_locations with lists...")
    print("=" * 70)
    try:
        test_sorted_division_locations_accepts_lists()
        print("=" * 70)
        print("All tests passed!")
    except Exception as e:
        print("=" * 70)
        print(f"Test failed with minimal example!")
        print(f"To reproduce, add this to your test:")
        # The test will fail on the first example, which will be the minimal case