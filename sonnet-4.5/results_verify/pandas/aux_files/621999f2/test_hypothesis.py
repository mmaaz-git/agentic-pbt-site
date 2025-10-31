from pandas.core.indexes.api import _get_combined_index
from hypothesis import given, strategies as st, settings, assume
import pandas as pd


@st.composite
def index_strategy(draw):
    data = draw(st.one_of(
        st.lists(st.integers(min_value=-100, max_value=100), min_size=0, max_size=15),
        st.lists(st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=5), min_size=0, max_size=15),
    ))
    return pd.Index(data)


@given(idx1=index_strategy(), idx2=index_strategy())
@settings(max_examples=300)
def test_combined_index_sort_actually_sorts(idx1, idx2):
    try:
        result = _get_combined_index([idx1, idx2], intersect=False, sort=True, copy=False)

        if len(result) > 0:
            assert result.is_monotonic_increasing or result.is_monotonic_decreasing, \
                f"sort=True should produce monotonic result: {result}"
    except (TypeError, ValueError) as e:
        assume(False)

# Run the test
print("Running Hypothesis test...")
print("=" * 60)

# First test with the specific failing input from the bug report
print("Testing the specific failing input from bug report:")
idx1 = pd.Index([0], dtype='int64')
idx2 = pd.Index(['a'], dtype='object')

print(f"idx1: {idx1} (dtype: {idx1.dtype})")
print(f"idx2: {idx2} (dtype: {idx2.dtype})")

try:
    result = _get_combined_index([idx1, idx2], intersect=False, sort=True, copy=False)
    print(f"Combined result: {result} (dtype: {result.dtype})")
    is_sorted = result.is_monotonic_increasing or result.is_monotonic_decreasing
    print(f"Is result sorted? {is_sorted}")
    if not is_sorted:
        print("FAILURE: Result is not sorted despite sort=True")
except (TypeError, ValueError) as e:
    print(f"Exception raised: {e}")

print()
print("=" * 60)
print("Running full Hypothesis test suite...")
try:
    test_combined_index_sort_actually_sorts()
    print("All Hypothesis tests passed!")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")