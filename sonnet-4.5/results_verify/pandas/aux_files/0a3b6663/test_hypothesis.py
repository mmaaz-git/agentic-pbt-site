import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from hypothesis import given, strategies as st, settings


@given(arr=st.lists(st.text(min_size=0, max_size=10), min_size=1, max_size=50))
@settings(max_examples=500)
def test_categorical_round_trip(arr):
    cat = pd.Categorical(arr)
    reconstructed = cat.categories[cat.codes]

    assert list(reconstructed) == arr, \
        f"Categorical round-trip failed: original {arr}, reconstructed {list(reconstructed)}"

if __name__ == "__main__":
    # Run with specific failing example
    test_arr = ['', '\x00']
    cat = pd.Categorical(test_arr)
    reconstructed = list(cat.categories[cat.codes])

    try:
        assert reconstructed == test_arr, \
            f"Categorical round-trip failed: original {test_arr}, reconstructed {reconstructed}"
        print(f"Test passed for {repr(test_arr)}")
    except AssertionError as e:
        print(f"Test failed: {e}")

    # Run full hypothesis test
    print("\nRunning full hypothesis test...")
    test_categorical_round_trip()