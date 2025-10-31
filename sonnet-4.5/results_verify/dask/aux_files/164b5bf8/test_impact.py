#!/usr/bin/env python3
"""Test the actual impact of the bug on filter extraction."""

import pandas as pd
import tempfile
import os
from pathlib import Path

# Create a simple test parquet file
def create_test_data():
    """Create a test parquet file."""
    df = pd.DataFrame({
        'col1': range(10),
        'col2': range(10, 20)
    })

    tmpdir = tempfile.mkdtemp()
    filepath = os.path.join(tmpdir, "test.parquet")
    df.to_parquet(filepath)
    return filepath, df

def test_filter_extraction():
    """Test if filters are properly extracted for normal vs reversed comparisons."""

    filepath, original_df = create_test_data()

    try:
        import dask.dataframe as dd
        from dask.dataframe.dask_expr.io.parquet import _DNF, ReadParquet
        from dask.dataframe.dask_expr._expr import GT, LT

        # Read the parquet file
        ddf = dd.read_parquet(filepath)

        # Test 1: Normal comparison (column > value)
        normal_result = ddf[ddf['col1'] > 5]

        # Test 2: Reversed comparison (value < column) - should be equivalent
        reversed_result = ddf[5 < ddf['col1']]

        print("Testing filter extraction:")
        print("=" * 50)

        # Check if both produce the same result
        normal_computed = normal_result.compute()
        reversed_computed = reversed_result.compute()

        print(f"Normal comparison (col1 > 5) result shape: {normal_computed.shape}")
        print(f"Reversed comparison (5 < col1) result shape: {reversed_computed.shape}")

        if normal_computed.equals(reversed_computed):
            print("✓ Both comparisons produce the same result")
        else:
            print("✗ Comparisons produce different results!")

        # Now let's check if filters are being extracted properly
        # This would require introspecting the expression tree, which is complex
        # But we can at least verify the basic behavior

        print("\nExpression tree inspection:")
        print(f"Normal expression: {ddf['col1'] > 5}")
        print(f"Reversed expression: {5 < ddf['col1']}")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        os.remove(filepath)
        os.rmdir(os.path.dirname(filepath))

if __name__ == "__main__":
    test_filter_extraction()