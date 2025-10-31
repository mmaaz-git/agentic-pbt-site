import pandas as pd
from hypothesis import given, strategies as st, settings
from dask.dataframe.io.io import sorted_division_locations


# Test the hypothesis property
@given(
    seq=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=100),
    chunksize=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=1000)
def test_sorted_division_locations_monotonic_locations(seq, chunksize):
    seq_pd = pd.Series(seq)
    divisions, locations = sorted_division_locations(seq_pd, chunksize=chunksize)

    for i in range(len(locations) - 1):
        assert locations[i] < locations[i+1], \
            f"Locations must be strictly increasing: locations[{i}]={locations[i]} >= locations[{i+1}]={locations[i+1]}"

# Test the specific failing case
print("Testing specific failing case:")
seq = pd.Series([0, 1, 0, 0])
divisions, locations = sorted_division_locations(seq, chunksize=1)

print(f"Input (unsorted): {list(seq)}")
print(f"Divisions: {divisions}")
print(f"Locations: {locations}")

# Check if locations are strictly increasing
strictly_increasing = all(locations[i] < locations[i+1] for i in range(len(locations) - 1))
print(f"Locations strictly increasing: {strictly_increasing}")

# Run hypothesis test to find failures
print("\nRunning hypothesis test...")
try:
    test_sorted_division_locations_monotonic_locations()
    print("Hypothesis test passed for 1000 examples")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")