import dask.dataframe as dd
import pandas as pd
from hypothesis import given, strategies as st, settings

# First test: Property-based test from the bug report
@given(
    st.lists(st.integers(min_value=-100, max_value=100), min_size=20, max_size=50),
    st.integers(min_value=2, max_value=5)
)
@settings(max_examples=50)
def test_head_tail_api_symmetry(data, npartitions):
    df = pd.DataFrame({'x': data})
    dask_df = dd.from_pandas(df, npartitions=npartitions)

    head_with_npartitions = dask_df.head(10, npartitions=-1)

    try:
        tail_with_npartitions = dask_df.tail(10, npartitions=-1)
        assert False, "tail() should have raised TypeError for npartitions parameter"
    except TypeError as e:
        assert "npartitions" in str(e) or "unexpected keyword argument" in str(e)
        print(f"✓ tail() correctly raised TypeError: {e}")

# Run the property-based test
print("Running property-based test...")
test_head_tail_api_symmetry()

print("\n" + "="*50 + "\n")

# Second test: Direct reproduction from the bug report
print("Direct reproduction test:")
df = pd.DataFrame({'x': range(100)})
dask_df = dd.from_pandas(df, npartitions=5)

# Test head with npartitions
try:
    head_result = dask_df.head(10, npartitions=-1)
    print(f"✓ head() with npartitions=-1: {len(head_result)} rows")
except Exception as e:
    print(f"✗ head() with npartitions=-1 failed: {e}")

# Test tail with npartitions (should fail)
try:
    tail_result = dask_df.tail(10, npartitions=-1)
    print(f"✗ tail() with npartitions=-1 unexpectedly succeeded: {len(tail_result)} rows")
except TypeError as e:
    print(f"✓ tail() with npartitions=-1 failed as expected: {e}")

# Test tail without npartitions (should work)
try:
    tail_result = dask_df.tail(10)
    print(f"✓ tail() without npartitions: {len(tail_result)} rows")
except Exception as e:
    print(f"✗ tail() without npartitions failed: {e}")

print("\n" + "="*50 + "\n")

# Let's also check the signatures
print("Checking method signatures:")
import inspect

print(f"head() signature: {inspect.signature(dask_df.head)}")
print(f"tail() signature: {inspect.signature(dask_df.tail)}")