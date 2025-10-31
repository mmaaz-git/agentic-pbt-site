#!/usr/bin/env python3
"""Property-based test from the bug report"""

from hypothesis import given, strategies as st, settings, assume
import pandas as pd
import dask.dataframe as dd
import traceback


@settings(max_examples=50)
@given(
    data=st.lists(st.integers(min_value=-100, max_value=100), min_size=5, max_size=20),
    n=st.integers(min_value=1, max_value=5)
)
def test_nlargest_nsmallest_disjoint(data, n):
    """nlargest and nsmallest should be disjoint"""
    assume(len(set(data)) >= 2 * n)

    df = pd.DataFrame({'x': data})
    ddf = dd.from_pandas(df, npartitions=2)

    largest = set(ddf.nlargest(n, 'x')['x'].compute())
    smallest = set(ddf.nsmallest(n, 'x')['x'].compute())

    assert len(largest & smallest) == 0


if __name__ == "__main__":
    print("Running property-based test...")
    print("=" * 60)

    # Test the specific failing input from the bug report
    print("Testing specific failing input: data=[0, 0, 0, 0, 1], n=1")
    data = [0, 0, 0, 0, 1]
    n = 1

    try:
        df = pd.DataFrame({'x': data})
        ddf = dd.from_pandas(df, npartitions=2)

        print(f"DataFrame: {df['x'].tolist()}")
        print(f"n={n}")

        print("\nAttempting to get nlargest...")
        largest = set(ddf.nlargest(n, 'x')['x'].compute())
        print(f"nlargest result: {largest}")

        print("\nAttempting to get nsmallest...")
        smallest = set(ddf.nsmallest(n, 'x')['x'].compute())
        print(f"nsmallest result: {smallest}")

        print(f"\nIntersection: {largest & smallest}")
        print(f"Length of intersection: {len(largest & smallest)}")

        if len(largest & smallest) == 0:
            print("✓ Test passed: nlargest and nsmallest are disjoint")
        else:
            print("✗ Test failed: nlargest and nsmallest are NOT disjoint")
    except Exception as e:
        print(f"\n✗ Test failed with exception: {type(e).__name__}: {e}")
        print("\nFull traceback:")
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Running full hypothesis test suite...")

    try:
        test_nlargest_nsmallest_disjoint()
        print("✓ All hypothesis tests passed!")
    except Exception as e:
        print(f"✗ Hypothesis tests failed: {e}")
        traceback.print_exc()