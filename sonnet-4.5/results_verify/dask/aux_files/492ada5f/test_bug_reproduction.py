import pandas as pd
import dask.dataframe as dd
from hypothesis import given, strategies as st, settings

# First, let's try the simple reproduction case
print("=" * 50)
print("Testing simple reproduction case")
print("=" * 50)

pdf = pd.DataFrame({'values': [1.0, 2.0, 3.0]})
ddf = dd.from_pandas(pdf, npartitions=2)

print("\nPandas version (expected behavior):")
pandas_result = pdf.nlargest(2, 'values')['values']
print(f"pdf.nlargest(2, 'values')['values'] = {pandas_result.tolist()}")

print("\nDask version (testing for bug):")
try:
    result = ddf.nlargest(2, 'values')['values'].compute()
    print(f"Success: {result.tolist()}")
except TypeError as e:
    print(f"Error: {e}")

print("\n" + "=" * 50)
print("Testing workarounds mentioned in bug report")
print("=" * 50)

print("\nWorkaround 1 - compute before selecting column:")
try:
    result1 = ddf.nlargest(2, 'values').compute()['values']
    print(f"Success: {result1.tolist()}")
except Exception as e:
    print(f"Error: {e}")

print("\nWorkaround 2 - use Series.nlargest directly:")
try:
    result2 = ddf['values'].nlargest(2).compute()
    print(f"Success: {result2.tolist()}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 50)
print("Testing property-based test")
print("=" * 50)

@given(
    st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=30
    )
)
@settings(max_examples=10)  # Reduced for testing
def test_nlargest_column_selection(values):
    n = min(5, len(values))

    pdf = pd.DataFrame({'values': values})
    ddf = dd.from_pandas(pdf, npartitions=2)

    try:
        result = ddf.nlargest(n, 'values')['values'].compute()
        expected = pdf.nlargest(n, 'values')['values']

        assert len(result) == len(expected)
        print(".", end="")
        return True
    except TypeError as e:
        print(f"\nFailed with TypeError: {e}")
        return False

print("\nRunning property-based test...")
try:
    test_nlargest_column_selection()
    print("\nAll tests passed!")
except Exception as e:
    print(f"\nTest failed with error: {e}")