"""Test file to reproduce the reset_index bug reported for Dask DataFrame"""

from hypothesis import given, strategies as st, assume, settings
import pandas as pd
import dask.dataframe as dd

# First, let's test with the property-based test
@settings(max_examples=300)
@given(
    data=st.lists(
        st.tuples(st.integers(-1000, 1000), st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)),
        min_size=1,
        max_size=100
    ),
    npartitions=st.integers(min_value=1, max_value=10)
)
def test_reset_index_matches_pandas(data, npartitions):
    pdf = pd.DataFrame(data, columns=['a', 'b'])
    assume(len(pdf) >= npartitions)

    ddf = dd.from_pandas(pdf, npartitions=npartitions)

    reset_ddf = ddf.reset_index(drop=True).compute()
    reset_pdf = pdf.reset_index(drop=True)

    pd.testing.assert_frame_equal(reset_ddf, reset_pdf, check_index_type=False)

# Run the property-based test
print("Running property-based test...")
try:
    test_reset_index_matches_pandas()
    print("Property-based test passed!")
except Exception as e:
    print(f"Property-based test failed: {e}")

# Now test the specific failing example
print("\n" + "="*60)
print("Testing specific failing example from bug report:")
print("="*60 + "\n")

data = [(0, 0.0), (0, 0.0)]
pdf = pd.DataFrame(data, columns=['a', 'b'])
ddf = dd.from_pandas(pdf, npartitions=2)

reset_pdf = pdf.reset_index(drop=True)
reset_ddf = ddf.reset_index(drop=True).compute()

print(f"Original pandas DataFrame:\n{pdf}")
print(f"\nOriginal pandas index: {pdf.index.tolist()}")

print(f"\nPandas after reset_index(drop=True):\n{reset_pdf}")
print(f"Pandas index after reset: {reset_pdf.index.tolist()}")

print(f"\nDask after reset_index(drop=True):\n{reset_ddf}")
print(f"Dask index after reset: {reset_ddf.index.tolist()}")

print("\nAssertion checks:")
print(f"Expected pandas index: [0, 1]")
print(f"Actual pandas index: {reset_pdf.index.tolist()}")
print(f"Actual dask index: {reset_ddf.index.tolist()}")

try:
    assert reset_pdf.index.tolist() == [0, 1], "Pandas index is not [0, 1]"
    print("✓ Pandas index assertion passed")
except AssertionError as e:
    print(f"✗ Pandas index assertion failed: {e}")

try:
    assert reset_ddf.index.tolist() == [0, 0], "Dask index is not [0, 0]"
    print("✓ Dask index matches bug report (showing [0, 0])")
except AssertionError as e:
    print(f"✗ Dask index assertion failed: {e}")

print("\nComparing pandas and dask results:")
try:
    pd.testing.assert_frame_equal(reset_ddf, reset_pdf, check_index_type=False)
    print("✓ DataFrames are equal")
except Exception as e:
    print(f"✗ DataFrames are NOT equal: {e}")