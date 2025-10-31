#!/usr/bin/env python3
"""Test the reported bug about empty chunks in pandas interchange protocol."""

import pandas as pd
import numpy as np
from pandas.core.interchange.dataframe import PandasDataFrameXchg

def test_reproduction():
    """Test the specific reproduction case from the bug report."""
    print("=" * 60)
    print("Testing reproduction case from bug report")
    print("=" * 60)

    df = pd.DataFrame({'A': [0, 1, 2, 3, 4]})
    interchange_df = df.__dataframe__()
    chunks = list(interchange_df.get_chunks(n_chunks=7))

    print(f"DataFrame has {len(df)} rows")
    print(f"Requested 7 chunks, got {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        num_rows = chunk.num_rows()
        print(f"Chunk {i}: {num_rows} rows")

    empty_count = sum(1 for c in chunks if c.num_rows() == 0)
    print(f"Empty chunks: {empty_count}")
    print()

    return empty_count

def test_hypothesis_case():
    """Test the hypothesis example from the bug report."""
    print("=" * 60)
    print("Testing hypothesis case (n_rows=5, n_chunks=7)")
    print("=" * 60)

    n_rows = 5
    n_chunks = 7

    df = pd.DataFrame(np.random.randn(n_rows, 3))
    xchg = PandasDataFrameXchg(df, allow_copy=True)
    chunks = list(xchg.get_chunks(n_chunks=n_chunks))

    print(f"DataFrame has {n_rows} rows")
    print(f"Requested {n_chunks} chunks, got {len(chunks)} chunks")

    empty_chunks = [c for c in chunks if c.num_rows() == 0]
    print(f"Found {len(empty_chunks)} empty chunks")

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk.num_rows()} rows")
    print()

    return len(empty_chunks)

def test_edge_cases():
    """Test additional edge cases."""
    print("=" * 60)
    print("Testing additional edge cases")
    print("=" * 60)

    # Test 1: n_chunks much larger than rows
    print("\nTest 1: 3 rows, 10 chunks")
    df = pd.DataFrame({'A': [0, 1, 2]})
    xchg = PandasDataFrameXchg(df, allow_copy=True)
    chunks = list(xchg.get_chunks(n_chunks=10))
    print(f"  Got {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk.num_rows()} rows")
    empty_count1 = sum(1 for c in chunks if c.num_rows() == 0)
    print(f"  Empty chunks: {empty_count1}")

    # Test 2: n_chunks equal to rows
    print("\nTest 2: 5 rows, 5 chunks")
    df = pd.DataFrame({'A': [0, 1, 2, 3, 4]})
    xchg = PandasDataFrameXchg(df, allow_copy=True)
    chunks = list(xchg.get_chunks(n_chunks=5))
    print(f"  Got {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk.num_rows()} rows")
    empty_count2 = sum(1 for c in chunks if c.num_rows() == 0)
    print(f"  Empty chunks: {empty_count2}")

    # Test 3: n_chunks less than rows
    print("\nTest 3: 10 rows, 3 chunks")
    df = pd.DataFrame({'A': range(10)})
    xchg = PandasDataFrameXchg(df, allow_copy=True)
    chunks = list(xchg.get_chunks(n_chunks=3))
    print(f"  Got {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {chunk.num_rows()} rows")
    empty_count3 = sum(1 for c in chunks if c.num_rows() == 0)
    print(f"  Empty chunks: {empty_count3}")

    print()

def test_column_chunks():
    """Test the same issue with PandasColumn.get_chunks()."""
    print("=" * 60)
    print("Testing PandasColumn.get_chunks()")
    print("=" * 60)

    from pandas.core.interchange.column import PandasColumn

    series = pd.Series([0, 1, 2, 3, 4])
    col = PandasColumn(series, allow_copy=True)
    chunks = list(col.get_chunks(n_chunks=7))

    print(f"Series has {len(series)} elements")
    print(f"Requested 7 chunks, got {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        size = chunk.size()
        print(f"Chunk {i}: {size} elements")

    empty_count = sum(1 for c in chunks if c.size() == 0)
    print(f"Empty chunks: {empty_count}")
    print()

    return empty_count

if __name__ == "__main__":
    # Run all tests
    empty1 = test_reproduction()
    empty2 = test_hypothesis_case()
    test_edge_cases()
    empty3 = test_column_chunks()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Reproduction case: {empty1} empty chunks")
    print(f"Hypothesis case: {empty2} empty chunks")
    print(f"Column case: {empty3} empty chunks")

    if empty1 > 0 or empty2 > 0 or empty3 > 0:
        print("\n✗ BUG CONFIRMED: Empty chunks are created when n_chunks > data size")
    else:
        print("\n✓ No bug found: No empty chunks created")