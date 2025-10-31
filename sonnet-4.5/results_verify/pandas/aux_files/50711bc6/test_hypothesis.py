import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings


@given(
    st.integers(min_value=2**53, max_value=2**60),
    st.integers(min_value=-1000, max_value=1000),
)
@settings(max_examples=100)
def test_grouped_diff_matches_ungrouped_diff(large_val, small_val):
    df = pd.DataFrame({
        'group': ['a', 'a'],
        'value': [large_val, small_val]
    })

    grouped_diff = df.groupby('group')['value'].diff()
    ungrouped_diff = df['value'].diff()

    assert ungrouped_diff.loc[1] == grouped_diff.loc[1], \
        f"Grouped diff ({grouped_diff.loc[1]}) != ungrouped diff ({ungrouped_diff.loc[1]}) for values {large_val}, {small_val}"

# Run the test
if __name__ == "__main__":
    try:
        test_grouped_diff_matches_ungrouped_diff()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")