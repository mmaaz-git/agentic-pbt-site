import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from pandas.api.interchange import from_dataframe
from hypothesis import given, strategies as st, settings
import traceback

@given(st.lists(st.integers(min_value=-(2**63), max_value=2**63-1), min_size=1, max_size=20))
@settings(max_examples=50)
def test_nullable_int_dtype(values):
    df = pd.DataFrame({'col': pd.array(values, dtype='Int64')})

    interchange_obj = df.__dataframe__()
    result = from_dataframe(interchange_obj)

    pd.testing.assert_frame_equal(result, df)

print("Running hypothesis test...")
try:
    test_nullable_int_dtype()
    print("All tests passed!")
except Exception as e:
    print(f"Test failed with error: {e}")
    traceback.print_exc()

# Try to find a simple failing case
print("\nManually testing some specific cases:")
test_cases = [
    [1, 2, 3],  # No NA values
    [1],  # Single value
    [None],  # Single NA
    [1, None],  # With NA
    [None, 1],  # NA first
    [1, None, 3],  # NA in middle
]

for i, values in enumerate(test_cases):
    print(f"\nTest case {i+1}: {values}")
    df = pd.DataFrame({'col': pd.array(values, dtype='Int64')})
    result = from_dataframe(df.__dataframe__())

    try:
        pd.testing.assert_frame_equal(result, df)
        print(f"  PASS - dtype preserved: {result['col'].dtype}")
    except AssertionError as e:
        print(f"  FAIL - Original: {df['col'].dtype}, Result: {result['col'].dtype}")
        if 'NA' in str(df['col'].values) or '<NA>' in str(df['col'].values):
            print(f"  Contains NA values: Yes")