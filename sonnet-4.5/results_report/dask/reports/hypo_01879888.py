from hypothesis import given, strategies as st, settings, example
from dask.dataframe.io.parquet.core import sorted_columns
import traceback

@given(
    st.lists(
        st.dictionaries(
            st.just("columns"),
            st.lists(
                st.fixed_dictionaries({
                    "name": st.text(min_size=1, max_size=10),
                    "min": st.one_of(st.none(), st.integers(), st.floats(allow_nan=False, allow_infinity=False)),
                    "max": st.one_of(st.none(), st.integers(), st.floats(allow_nan=False, allow_infinity=False)),
                }),
                min_size=1,
                max_size=5
            ),
            min_size=1,
            max_size=1
        ),
        min_size=1,
        max_size=10
    )
)
@example(statistics=[{'columns': [{'name': '0', 'min': 0, 'max': None}]}])  # The failing example
@settings(max_examples=100)
def test_sorted_columns_divisions_are_sorted(statistics):
    """Test that sorted_columns returns properly sorted divisions without crashing"""
    try:
        result = sorted_columns(statistics)
        # If we get a result, verify the divisions are sorted
        for item in result:
            divisions = item["divisions"]
            # Check that divisions are sorted (this should always be true per the assertion in the function)
            assert divisions == sorted(divisions), f"Divisions not sorted: {divisions}"
    except TypeError as e:
        # If we hit the bug, print details
        print(f"\nFailing input: {statistics}")
        print(f"TypeError: {e}")
        traceback.print_exc()
        raise

# Run the test
if __name__ == "__main__":
    print("Running property-based test for sorted_columns...")
    try:
        test_sorted_columns_divisions_are_sorted()
        print("Test completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")