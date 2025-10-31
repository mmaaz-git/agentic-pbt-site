import pandas as pd
import dask.dataframe as dd
from hypothesis import given, strategies as st, settings

# First, test the specific example from the bug report
print("=== Testing specific example ===")
pdf = pd.DataFrame({'col': ['hello', 1, None]})
print("Original:", pdf['col'].tolist())
print("Original types:", [type(x).__name__ for x in pdf['col']])
print("Original dtype:", pdf['col'].dtype)

ddf = dd.from_pandas(pdf, npartitions=2)
result = ddf.compute()
print("\nResult:", result['col'].tolist())
print("Result types:", [type(x).__name__ for x in result['col']])
print("Result dtype:", result['col'].dtype)

print("\n=== Assertion tests ===")
try:
    assert pdf['col'].iloc[1] == 1
    print("✓ Original has integer 1 at position 1")
except AssertionError:
    print("✗ Original does not have integer 1 at position 1")

try:
    assert isinstance(pdf['col'].iloc[1], int)
    print("✓ Original value at position 1 is int type")
except AssertionError:
    print("✗ Original value at position 1 is not int type")

try:
    assert result['col'].iloc[1] == '1'
    print("✓ Result has string '1' at position 1")
except AssertionError:
    print("✗ Result does not have string '1' at position 1")

try:
    assert isinstance(result['col'].iloc[1], str)
    print("✓ Result value at position 1 is str type")
except AssertionError:
    print("✗ Result value at position 1 is not str type")

print("\n=== Checking equality ===")
try:
    pd.testing.assert_frame_equal(pdf, result)
    print("✓ DataFrames are equal")
except Exception as e:
    print(f"✗ DataFrames are NOT equal: {e}")

# Run the hypothesis test
print("\n=== Running hypothesis test ===")
test_failures = []

@given(
    st.data(),
    st.integers(min_value=0, max_value=100),
    st.integers(min_value=1, max_value=5)
)
@settings(max_examples=100)
def test_from_pandas_compute_roundtrip(data, n_rows, n_cols):
    if n_rows == 0:
        return

    df_dict = {}
    for i in range(n_cols):
        col_name = f'col_{i}'
        df_dict[col_name] = data.draw(
            st.lists(
                st.one_of(
                    st.integers(),
                    st.floats(allow_nan=False, allow_infinity=False),
                    st.text(max_size=10)
                ),
                min_size=n_rows,
                max_size=n_rows
            )
        )

    pdf = pd.DataFrame(df_dict)
    ddf = dd.from_pandas(pdf, npartitions=2)
    result = ddf.compute()

    try:
        pd.testing.assert_frame_equal(pdf, result)
    except Exception as e:
        test_failures.append({
            'pdf': pdf,
            'result': result,
            'error': str(e)
        })
        raise

try:
    test_from_pandas_compute_roundtrip()
    print("All hypothesis tests passed")
except Exception as e:
    print(f"Hypothesis test failed: {e}")
    if test_failures:
        failure = test_failures[0]
        print(f"\nFirst failure example:")
        print(f"Original PDF:\n{failure['pdf']}")
        print(f"\nResult:\n{failure['result']}")
        print(f"\nError: {failure['error']}")