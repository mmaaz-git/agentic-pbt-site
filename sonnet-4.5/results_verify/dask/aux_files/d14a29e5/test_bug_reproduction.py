#!/usr/bin/env python3
"""Reproduce the reported bug in dask.dataframe.io.parquet.utils._normalize_index_columns"""

# First, let's try the simple reproduction case
from dask.dataframe.io.parquet.utils import _normalize_index_columns

print("=== Simple Reproduction Case ===")
user_columns = None
data_columns = ['0']
user_index = None
data_index = ['0']

try:
    column_names, index_names = _normalize_index_columns(
        user_columns, data_columns, user_index, data_index
    )

    print(f"column_names: {column_names}")
    print(f"index_names: {index_names}")
    print(f"Intersection: {set(column_names).intersection(set(index_names))}")
    print(f"Has overlap: {len(set(column_names).intersection(set(index_names))) > 0}")
except Exception as e:
    print(f"Error occurred: {e}")

# Now let's run the hypothesis test
print("\n=== Hypothesis Test ===")
from hypothesis import given, strategies as st, settings
import traceback

@given(
    st.one_of(st.none(), st.text(min_size=1, max_size=10), st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5)),
    st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5),
    st.one_of(
        st.none(),
        st.just(False),
        st.text(min_size=1, max_size=10),
        st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5)
    ),
    st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5),
)
@settings(max_examples=100)
def test_normalize_index_columns_no_intersection(user_columns, data_columns, user_index, data_index):
    try:
        column_names, index_names = _normalize_index_columns(
            user_columns, data_columns, user_index, data_index
        )
        intersection = set(column_names).intersection(set(index_names))
        assert len(intersection) == 0, f"Found intersection: {intersection}"
    except ValueError as e:
        if "must not intersect" in str(e):
            pass  # This is expected behavior
        else:
            raise

try:
    test_normalize_index_columns_no_intersection()
    print("All hypothesis tests passed")
except AssertionError as e:
    print(f"Hypothesis test failed with assertion error: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"Hypothesis test failed with exception: {e}")
    traceback.print_exc()

# Test the specific case mentioned in the bug report
print("\n=== Testing specific failing input from bug report ===")
try:
    column_names, index_names = _normalize_index_columns(
        None, ['0'], None, ['0']
    )
    intersection = set(column_names).intersection(set(index_names))
    print(f"With user_columns=None, data_columns=['0'], user_index=None, data_index=['0']:")
    print(f"  Result: columns={column_names}, index={index_names}")
    print(f"  Intersection: {intersection}")
    if intersection:
        print(f"  BUG CONFIRMED: Found overlap when both should be distinct")
except Exception as e:
    print(f"Error: {e}")

# Let's also test what happens when user specifies overlapping columns/indices
print("\n=== Testing user-specified overlap (should raise ValueError) ===")
try:
    column_names, index_names = _normalize_index_columns(
        ['0'], ['0', '1'], ['0'], ['0', '2']
    )
    print(f"With overlapping user specifications:")
    print(f"  Result: columns={column_names}, index={index_names}")
    print(f"  No error raised - this might be unexpected")
except ValueError as e:
    print(f"ValueError raised as expected: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")