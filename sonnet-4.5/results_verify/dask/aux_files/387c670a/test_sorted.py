import pandas as pd
from dask.dataframe.io.io import sorted_division_locations

# Test with properly sorted inputs as expected by the function
print("Testing with properly sorted input:")
seq_sorted = pd.Series([0, 0, 0, 1])
divisions, locations = sorted_division_locations(seq_sorted, chunksize=1)

print(f"Input (sorted): {list(seq_sorted)}")
print(f"Divisions: {divisions}")
print(f"Locations: {locations}")

# Check if locations are strictly increasing
strictly_increasing = all(locations[i] < locations[i+1] for i in range(len(locations) - 1))
print(f"Locations strictly increasing: {strictly_increasing}")

print("\n" + "="*50 + "\n")

# Test with the example from docstring
print("Testing with docstring example:")
L = ['A', 'B', 'C', 'D', 'E', 'F']
divisions, locations = sorted_division_locations(L, chunksize=2)
print(f"Input: {L}")
print(f"Divisions: {divisions}")
print(f"Locations: {locations}")
print(f"Expected: (['A', 'C', 'E', 'F'], [0, 2, 4, 6])")
print(f"Match: {(divisions, locations) == (['A', 'C', 'E', 'F'], [0, 2, 4, 6])}")

print("\n" + "="*50 + "\n")

# Test with duplicates (from docstring)
print("Testing with duplicates example from docstring:")
L = ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'C']
divisions, locations = sorted_division_locations(L, chunksize=3)
print(f"Input: {L}")
print(f"Divisions: {divisions}")
print(f"Locations: {locations}")
print(f"Expected: (['A', 'B', 'C', 'C'], [0, 4, 7, 8])")
print(f"Match: {(divisions, locations) == (['A', 'B', 'C', 'C'], [0, 4, 7, 8])}")