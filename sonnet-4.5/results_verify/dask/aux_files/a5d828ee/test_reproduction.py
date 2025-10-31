"""Test reproduction of the sorted_division_locations bug"""

# First, let's try the examples from the docstring
from dask.dataframe.io.io import sorted_division_locations

print("Testing example from docstring:")
try:
    L = ['A', 'B', 'C', 'D', 'E', 'F']
    print(f"Input: L = {L}, chunksize=2")
    divisions, locations = sorted_division_locations(L, chunksize=2)
    print(f"Success! Result: divisions={divisions}, locations={locations}")
except Exception as e:
    print(f"ERROR: {e}")

print("\nTesting second example from docstring:")
try:
    L = ['A', 'B', 'C', 'D', 'E', 'F']
    print(f"Input: L = {L}, chunksize=3")
    divisions, locations = sorted_division_locations(L, chunksize=3)
    print(f"Success! Result: divisions={divisions}, locations={locations}")
except Exception as e:
    print(f"ERROR: {e}")

print("\nTesting with the hypothesis test example:")
try:
    seq = [0]
    print(f"Input: seq = {seq}, chunksize=1")
    divisions, locations = sorted_division_locations(seq, chunksize=1)
    print(f"Success! Result: divisions={divisions}, locations={locations}")
    assert locations[0] == 0
    assert locations[-1] == len(seq)
    print("Assertions passed!")
except Exception as e:
    print(f"ERROR: {e}")

# Now let's test with numpy arrays to see if those work
import numpy as np
print("\nTesting with numpy array:")
try:
    arr = np.array(['A', 'B', 'C', 'D', 'E', 'F'])
    print(f"Input: numpy array {arr}, chunksize=2")
    divisions, locations = sorted_division_locations(arr, chunksize=2)
    print(f"Success! Result: divisions={divisions}, locations={locations}")
except Exception as e:
    print(f"ERROR: {e}")