#!/usr/bin/env python3
"""Test the bug reported for sorted_division_locations"""

import traceback

print("Testing the bug report for sorted_division_locations")
print("=" * 60)

# Test 1: Try the example from the docstring
print("\n1. Testing example from the docstring:")
print("Code: sorted_division_locations(['A', 'B', 'C', 'D', 'E', 'F'], chunksize=2)")
try:
    from dask.dataframe.io.io import sorted_division_locations
    L = ['A', 'B', 'C', 'D', 'E', 'F']
    result = sorted_division_locations(L, chunksize=2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    traceback.print_exc()

# Test 2: Try the minimal failing case from bug report
print("\n2. Testing minimal failing case from bug report:")
print("Code: sorted_division_locations(['A'], chunksize=2)")
try:
    from dask.dataframe.io.io import sorted_division_locations
    seq = ['A']
    divisions, locations = sorted_division_locations(seq, chunksize=2)
    print(f"divisions={divisions}, locations={locations}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test 3: Try with numpy array (should work)
print("\n3. Testing with numpy array (expected to work):")
print("Code: sorted_division_locations(np.array(['A', 'B', 'C', 'D', 'E', 'F']), chunksize=2)")
try:
    import numpy as np
    from dask.dataframe.io.io import sorted_division_locations
    L = np.array(['A', 'B', 'C', 'D', 'E', 'F'])
    result = sorted_division_locations(L, chunksize=2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test 4: Try with pandas Series (should work)
print("\n4. Testing with pandas Series (expected to work):")
print("Code: sorted_division_locations(pd.Series(['A', 'B', 'C', 'D', 'E', 'F']), chunksize=2)")
try:
    import pandas as pd
    from dask.dataframe.io.io import sorted_division_locations
    L = pd.Series(['A', 'B', 'C', 'D', 'E', 'F'])
    result = sorted_division_locations(L, chunksize=2)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Test 5: Run the hypothesis test
print("\n5. Running the hypothesis test from the bug report:")
try:
    from hypothesis import given, strategies as st
    from dask.dataframe.io.io import sorted_division_locations

    @given(seq=st.lists(st.text(alphabet='ABC', min_size=1, max_size=1), min_size=1, max_size=10))
    def test_sorted_division_locations_accepts_python_lists(seq):
        """
        Property: sorted_division_locations should accept plain Python lists.

        The function's docstring provides multiple examples using plain Python lists:
        - L = ['A', 'B', 'C', 'D', 'E', 'F']
        - sorted_division_locations(L, chunksize=2)

        This property verifies the documented behavior works.
        """
        divisions, locations = sorted_division_locations(seq, chunksize=2)

        assert isinstance(divisions, list)
        assert isinstance(locations, list)
        assert locations[0] == 0
        assert locations[-1] == len(seq)

    # Run the test
    test_sorted_division_locations_accepts_python_lists()
    print("Hypothesis test passed!")

except Exception as e:
    print(f"Hypothesis test failed: {type(e).__name__}: {e}")
    # Try to get the failing example
    import sys
    if hasattr(e, '__notes__'):
        for note in e.__notes__:
            print(f"Note: {note}")