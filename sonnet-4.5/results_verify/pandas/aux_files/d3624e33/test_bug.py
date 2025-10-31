#!/usr/bin/env python3
"""Test the reported bug with PyArrow engine single-value CSV parsing."""

from io import StringIO
import pandas as pd
import traceback

print("=" * 60)
print("Testing single-value CSV parsing with different engines")
print("=" * 60)

csv_content = "0"
print(f"CSV content: '{csv_content}'")
print(f"CSV content bytes: {csv_content.encode()}")
print(f"CSV content has trailing newline: {csv_content.endswith('\\n')}")
print()

# Test C engine
print("Testing C engine:")
try:
    df_c = pd.read_csv(StringIO(csv_content), header=None, engine='c')
    print("✓ C engine result:")
    print(df_c)
    print(f"Shape: {df_c.shape}")
    print(f"Value: {df_c.iloc[0, 0]}")
    print(f"Type: {type(df_c.iloc[0, 0])}")
except Exception as e:
    print(f"✗ C engine failed with: {e}")
    traceback.print_exc()
print()

# Test Python engine
print("Testing Python engine:")
try:
    df_python = pd.read_csv(StringIO(csv_content), header=None, engine='python')
    print("✓ Python engine result:")
    print(df_python)
    print(f"Shape: {df_python.shape}")
    print(f"Value: {df_python.iloc[0, 0]}")
    print(f"Type: {type(df_python.iloc[0, 0])}")
except Exception as e:
    print(f"✗ Python engine failed with: {e}")
    traceback.print_exc()
print()

# Test PyArrow engine
print("Testing PyArrow engine:")
try:
    df_pyarrow = pd.read_csv(StringIO(csv_content), header=None, engine='pyarrow')
    print("✓ PyArrow engine result:")
    print(df_pyarrow)
    print(f"Shape: {df_pyarrow.shape}")
    print(f"Value: {df_pyarrow.iloc[0, 0]}")
    print(f"Type: {type(df_pyarrow.iloc[0, 0])}")
except Exception as e:
    print(f"✗ PyArrow engine failed with: {e}")
    traceback.print_exc()
print()

# Test other single values
print("=" * 60)
print("Testing other single values:")
print("=" * 60)

test_values = ["42", "hello", "3.14", "-1"]
for val in test_values:
    print(f"\nTesting value: '{val}'")
    for engine in ['c', 'python', 'pyarrow']:
        try:
            df = pd.read_csv(StringIO(val), header=None, engine=engine)
            print(f"  {engine:8s} engine: ✓ Shape {df.shape}")
        except Exception as e:
            print(f"  {engine:8s} engine: ✗ {type(e).__name__}: {str(e)[:50]}...")

# Test with trailing newline
print("\n" + "=" * 60)
print("Testing with trailing newline:")
print("=" * 60)

csv_with_newline = "0\n"
print(f"CSV content: '{csv_with_newline.rstrip()}'\\n")
for engine in ['c', 'python', 'pyarrow']:
    try:
        df = pd.read_csv(StringIO(csv_with_newline), header=None, engine=engine)
        print(f"  {engine:8s} engine: ✓ Shape {df.shape}")
    except Exception as e:
        print(f"  {engine:8s} engine: ✗ {e}")

# Run the hypothesis test
print("\n" + "=" * 60)
print("Running hypothesis test:")
print("=" * 60)

from hypothesis import given, strategies as st, assume, settings
import pandas.io.parsers as parsers

@settings(max_examples=100)
@given(
    data=st.lists(
        st.lists(st.integers(0, 100), min_size=1, max_size=5),
        min_size=1,
        max_size=20
    )
)
def test_engine_equivalence_with_pyarrow(data):
    num_cols = len(data[0])
    assume(all(len(row) == num_cols for row in data))

    csv_content = "\n".join(",".join(str(val) for val in row) for row in data)

    df_c = parsers.read_csv(StringIO(csv_content), header=None, engine='c')
    df_pyarrow = parsers.read_csv(StringIO(csv_content), header=None, engine='pyarrow')

    pd.testing.assert_frame_equal(df_c, df_pyarrow, check_dtype=False)

try:
    test_engine_equivalence_with_pyarrow()
    print("✗ Hypothesis test passed (no failures found)")
except Exception as e:
    print(f"✓ Hypothesis test found failure: {e}")
    print(f"This confirms the bug exists")