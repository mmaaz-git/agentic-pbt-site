import pandas as pd
from dask.dataframe.utils import _maybe_sort

# First, test the simple reproduction case
print("=== Simple Reproduction Test ===")
df = pd.DataFrame(
    {'A': [2, 1], 'B': [4, 3]},
    index=pd.Index([10, 20], name='A')
)

print(f"Before: df.index.names = {df.index.names}")

result = _maybe_sort(df, check_index=True)

print(f"After: result.index.names = {result.index.names}")
print(f"Expected: ['A'], Actual: {result.index.names}")
print(f"Bug reproduced: {result.index.names[0] != 'A'}")

# Now test the property-based test
print("\n=== Property-Based Test ===")
from hypothesis import given, strategies as st

@given(st.lists(st.integers(), min_size=2, max_size=10))
def test_maybe_sort_preserves_index_names(data):
    df = pd.DataFrame({'A': data}, index=pd.Index(range(len(data)), name='A'))
    original_name = df.index.names[0]

    result = _maybe_sort(df, check_index=True)

    assert result.index.names[0] == original_name, \
        f"Index name changed from {original_name} to {result.index.names[0]}"

# Run the property test
try:
    test_maybe_sort_preserves_index_names()
    print("Property test passed (unexpected)")
except AssertionError as e:
    print(f"Property test failed as expected: {e}")
except Exception as e:
    print(f"Property test raised unexpected error: {e}")