import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

# Test 1: Run the example from the docstring
print("Test 1: Running docstring example...")
try:
    from dask.dataframe.io.io import sorted_division_locations

    L = ['A', 'B', 'C', 'D', 'E', 'F']
    result = sorted_division_locations(L, chunksize=2)
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Test 2: Run the hypothesis test
print("Test 2: Running hypothesis test...")
from hypothesis import given, strategies as st, settings

@given(
    seq=st.lists(st.integers(min_value=0, max_value=100), min_size=1, max_size=100),
    chunksize=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=10)
def test_sorted_division_locations_basic_invariants(seq, chunksize):
    seq_sorted = sorted(seq)
    try:
        divisions, locations = sorted_division_locations(seq_sorted, chunksize=chunksize)
        assert len(divisions) == len(locations)
        assert locations[0] == 0
        assert locations[-1] == len(seq_sorted)
        return True
    except TypeError as e:
        if "No dispatch for <class 'list'>" in str(e):
            print(f"  Failed with list {seq_sorted[:5]}... (len={len(seq_sorted)}), chunksize={chunksize}")
            raise
        else:
            raise

try:
    test_sorted_division_locations_basic_invariants()
    print("All hypothesis tests passed!")
except Exception as e:
    print(f"Hypothesis test failed: {type(e).__name__}")

print("\n" + "="*50 + "\n")

# Test 3: Check if numpy arrays work
print("Test 3: Testing with numpy array...")
import numpy as np

try:
    arr = np.array(['A', 'B', 'C', 'D', 'E', 'F'])
    result = sorted_division_locations(arr, chunksize=2)
    print(f"Success with numpy array! Result: {result}")
except Exception as e:
    print(f"Error with numpy array: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Test 4: Testing with pandas Series
print("Test 4: Testing with pandas Series...")
import pandas as pd

try:
    s = pd.Series(['A', 'B', 'C', 'D', 'E', 'F'])
    result = sorted_division_locations(s, chunksize=2)
    print(f"Success with pandas Series! Result: {result}")
except Exception as e:
    print(f"Error with pandas Series: {type(e).__name__}: {e}")