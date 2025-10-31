import tempfile
import os
import pandas as pd
from hypothesis import given, strategies as st, settings
from pandas.testing import assert_frame_equal


@settings(max_examples=100, deadline=None)
@given(
    data=st.lists(
        st.lists(
            st.one_of(
                st.integers(min_value=-1e6, max_value=1e6),
                st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6),
                st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=0, max_size=20),
                st.booleans(),
            ),
            min_size=1,
            max_size=10,
        ),
        min_size=1,
        max_size=20,
    )
)
def test_round_trip_basic(data):
    if not all(len(row) == len(data[0]) for row in data):
        return

    df_original = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        df_original.to_excel(tmp_path, index=False)
        df_read = pd.read_excel(tmp_path)

        assert_frame_equal(df_original, df_read, check_dtype=False)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

# Test the specific failing case
if __name__ == "__main__":
    # Test with the specific failing input mentioned
    data = [['']]
    print(f"Testing with data: {data}")

    df_original = pd.DataFrame(data)
    print(f"Original DataFrame:\n{df_original}")
    print(f"Original values: {df_original.values.tolist()}")

    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        df_original.to_excel(tmp_path, index=False)
        df_read = pd.read_excel(tmp_path)

        print(f"\nDataFrame after round-trip:\n{df_read}")
        print(f"Values after round-trip: {df_read.values.tolist()}")

        try:
            assert_frame_equal(df_original, df_read, check_dtype=False)
            print("\nAssertion PASSED - DataFrames are equal")
        except AssertionError as e:
            print(f"\nAssertion FAILED - DataFrames are NOT equal:")
            print(e)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Also run the full hypothesis test
    print("\n" + "="*50)
    print("Running Hypothesis test...")
    test_round_trip_basic()