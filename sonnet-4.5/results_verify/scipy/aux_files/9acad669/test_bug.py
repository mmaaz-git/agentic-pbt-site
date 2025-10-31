#!/usr/bin/env python3
"""Test script to reproduce the floyd_warshall bug"""

import numpy as np
from scipy.sparse.csgraph import floyd_warshall
import sys

def test_reproduction():
    """Test the exact reproduction case from the bug report"""
    print("=" * 60)
    print("REPRODUCTION TEST FROM BUG REPORT")
    print("=" * 60)

    np.random.seed(0)
    graph = np.random.rand(3, 3) * 10
    graph_f = np.asfortranarray(graph, dtype=np.float64)

    print("Original graph (F-contiguous):")
    print(graph_f)
    print(f"Is F-contiguous: {graph_f.flags['F_CONTIGUOUS']}")
    print(f"Is C-contiguous: {graph_f.flags['C_CONTIGUOUS']}")
    print(f"Diagonal: {np.diag(graph_f)}")

    # Capture stderr
    import warnings
    warnings.filterwarnings('error')

    try:
        result = floyd_warshall(graph_f, directed=True, overwrite=True)
    except Exception as e:
        print(f"\nException raised: {e}")
        return

    print("\nResult after floyd_warshall:")
    print(result)
    print(f"Diagonal: {np.diag(result)}")
    print(f"\nExpected diagonal: [0, 0, 0]")
    print(f"Is result identical to input? {np.array_equal(result, graph_f)}")

    # Check if diagonal is zero (which it should be for shortest paths)
    if not np.allclose(np.diag(result), 0):
        print("\nBUG CONFIRMED: Diagonal should be zero for shortest paths!")
    else:
        print("\nNo bug: Diagonal is correctly zero")

def test_hypothesis():
    """Test the hypothesis-based property test"""
    print("\n" + "=" * 60)
    print("HYPOTHESIS PROPERTY TEST")
    print("=" * 60)

    from hypothesis import given, strategies as st
    import numpy as np
    from scipy.sparse.csgraph import floyd_warshall

    failed = False
    failure_example = None

    @given(st.integers(min_value=3, max_value=12))
    def test_floyd_warshall_overwrite_consistency(n):
        nonlocal failed, failure_example
        graph = np.random.rand(n, n) * 10
        graph_c = np.ascontiguousarray(graph, dtype=np.float64)
        graph_f = np.asfortranarray(graph, dtype=np.float64)

        dist_c = floyd_warshall(graph_c, directed=True, overwrite=False)
        dist_f = floyd_warshall(graph_f, directed=True, overwrite=True)

        try:
            np.testing.assert_allclose(dist_c, dist_f, rtol=1e-10, atol=1e-10)
        except AssertionError as e:
            if not failed:
                failed = True
                failure_example = n
                print(f"Property test failed for n={n}")
                print(f"C-contiguous result diagonal: {np.diag(dist_c)[:3]}")
                print(f"F-contiguous result diagonal: {np.diag(dist_f)[:3]}")
            raise

    try:
        test_floyd_warshall_overwrite_consistency()
    except:
        print(f"Property test FAILED (first failure at n={failure_example})")
    else:
        print("Property test PASSED")

def test_c_contiguous_comparison():
    """Compare C-contiguous vs F-contiguous arrays"""
    print("\n" + "=" * 60)
    print("C-CONTIGUOUS VS F-CONTIGUOUS COMPARISON")
    print("=" * 60)

    np.random.seed(42)
    graph = np.random.rand(4, 4) * 10

    # Test with C-contiguous
    graph_c = np.ascontiguousarray(graph, dtype=np.float64)
    print("Testing C-contiguous array:")
    print(f"Is C-contiguous: {graph_c.flags['C_CONTIGUOUS']}")
    result_c = floyd_warshall(graph_c.copy(), directed=True, overwrite=True)
    print(f"C-contiguous diagonal after floyd_warshall: {np.diag(result_c)}")

    # Test with F-contiguous
    graph_f = np.asfortranarray(graph, dtype=np.float64)
    print("\nTesting F-contiguous array:")
    print(f"Is F-contiguous: {graph_f.flags['F_CONTIGUOUS']}")
    result_f = floyd_warshall(graph_f.copy(), directed=True, overwrite=True)
    print(f"F-contiguous diagonal after floyd_warshall: {np.diag(result_f)}")

    # Test without overwrite
    print("\nTesting F-contiguous without overwrite:")
    result_f_no_overwrite = floyd_warshall(graph_f, directed=True, overwrite=False)
    print(f"F-contiguous diagonal (no overwrite): {np.diag(result_f_no_overwrite)}")

    # Compare
    print("\nComparison:")
    print(f"Are C and F results equal? {np.allclose(result_c, result_f)}")
    print(f"Are F with/without overwrite equal? {np.allclose(result_f, result_f_no_overwrite)}")

def test_what_floyd_warshall_should_do():
    """Test what Floyd-Warshall algorithm should actually compute"""
    print("\n" + "=" * 60)
    print("FLOYD-WARSHALL EXPECTED BEHAVIOR")
    print("=" * 60)

    # Simple 3-node graph
    graph = np.array([[0, 5, np.inf],
                      [np.inf, 0, 3],
                      [2, np.inf, 0]], dtype=np.float64)

    print("Input graph (0 = same node, inf = no direct edge):")
    print(graph)

    result = floyd_warshall(graph, directed=True, overwrite=False)
    print("\nExpected shortest paths:")
    print("- Node 0 to 0: 0 (same node)")
    print("- Node 0 to 1: 5 (direct)")
    print("- Node 0 to 2: 8 (via node 1: 0->1->2)")
    print("- Node 1 to 0: 5 (via node 2: 1->2->0)")
    print("- Node 1 to 1: 0 (same node)")
    print("- Node 1 to 2: 3 (direct)")
    print("- Node 2 to 0: 2 (direct)")
    print("- Node 2 to 1: 7 (via node 0: 2->0->1)")
    print("- Node 2 to 2: 0 (same node)")

    print("\nActual result:")
    print(result)
    print(f"\nDiagonal (should be all zeros): {np.diag(result)}")

if __name__ == "__main__":
    test_reproduction()
    test_hypothesis()
    test_c_contiguous_comparison()
    test_what_floyd_warshall_should_do()