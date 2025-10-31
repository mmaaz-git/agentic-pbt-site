#!/usr/bin/env python3
"""Detailed test to understand the bug behavior"""

import numpy as np
from scipy.sparse.csgraph import floyd_warshall
import sys
import warnings

def test_detailed():
    """Test with detailed output"""
    print("=" * 60)
    print("DETAILED INVESTIGATION")
    print("=" * 60)

    # Test 1: Simple case
    np.random.seed(0)
    graph = np.random.rand(3, 3) * 10

    # Test with C-contiguous
    print("\n1. C-contiguous with overwrite=True:")
    graph_c = np.ascontiguousarray(graph, dtype=np.float64)
    graph_c_copy = graph_c.copy()
    result_c = floyd_warshall(graph_c, directed=True, overwrite=True)
    print(f"   Input modified? {not np.array_equal(graph_c, graph_c_copy)}")
    print(f"   Result is input? {result_c is graph_c}")
    print(f"   Diagonal: {np.diag(result_c)}")

    # Test with F-contiguous
    print("\n2. F-contiguous with overwrite=True:")
    graph_f = np.asfortranarray(graph, dtype=np.float64)
    graph_f_copy = graph_f.copy()
    result_f = floyd_warshall(graph_f, directed=True, overwrite=True)
    print(f"   Input modified? {not np.array_equal(graph_f, graph_f_copy)}")
    print(f"   Result is input? {result_f is graph_f}")
    print(f"   Diagonal: {np.diag(result_f)}")

    # Test F-contiguous without overwrite
    print("\n3. F-contiguous with overwrite=False:")
    graph_f2 = np.asfortranarray(graph, dtype=np.float64)
    graph_f2_copy = graph_f2.copy()
    result_f2 = floyd_warshall(graph_f2, directed=True, overwrite=False)
    print(f"   Input modified? {not np.array_equal(graph_f2, graph_f2_copy)}")
    print(f"   Result is input? {result_f2 is graph_f2}")
    print(f"   Diagonal: {np.diag(result_f2)}")

    # Test C-contiguous without overwrite
    print("\n4. C-contiguous with overwrite=False:")
    graph_c2 = np.ascontiguousarray(graph, dtype=np.float64)
    graph_c2_copy = graph_c2.copy()
    result_c2 = floyd_warshall(graph_c2, directed=True, overwrite=False)
    print(f"   Input modified? {not np.array_equal(graph_c2, graph_c2_copy)}")
    print(f"   Result is input? {result_c2 is graph_c2}")
    print(f"   Diagonal: {np.diag(result_c2)}")

    # Test F-contiguous with different dtype
    print("\n5. F-contiguous float32 with overwrite=True:")
    graph_f32 = np.asfortranarray(graph, dtype=np.float32)
    result_f32 = floyd_warshall(graph_f32, directed=True, overwrite=True)
    print(f"   Diagonal: {np.diag(result_f32)}")

    # Check if silently falls back to non-overwrite
    print("\n6. Checking if F-contiguous silently ignores overwrite:")
    graph_f3 = np.asfortranarray(graph, dtype=np.float64)
    id_before = id(graph_f3)
    result_f3 = floyd_warshall(graph_f3, directed=True, overwrite=True)
    id_after = id(result_f3)
    print(f"   ID before: {id_before}")
    print(f"   ID after:  {id_after}")
    print(f"   Same object? {id_before == id_after}")
    print(f"   Arrays equal? {np.array_equal(graph_f3, result_f3)}")

if __name__ == "__main__":
    # Suppress warnings temporarily to see cleaner output
    warnings.filterwarnings('ignore')
    test_detailed()