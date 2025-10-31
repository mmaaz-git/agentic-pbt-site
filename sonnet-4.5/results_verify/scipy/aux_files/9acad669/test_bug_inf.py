#!/usr/bin/env python3
"""Test to understand the inf behavior"""

import numpy as np
from scipy.sparse.csgraph import floyd_warshall
import warnings

def test_inf_behavior():
    """Test what's happening with inf"""
    print("=" * 60)
    print("UNDERSTANDING INF BEHAVIOR")
    print("=" * 60)

    # Create a simple graph
    np.random.seed(0)
    graph = np.array([[0, 2, 3],
                      [2, 0, 1],
                      [3, 1, 0]], dtype=np.float64)

    print("Input graph (symmetric, simple):")
    print(graph)

    # Test with C-contiguous
    print("\n1. C-contiguous with overwrite=True:")
    graph_c = np.ascontiguousarray(graph, dtype=np.float64)
    result_c = floyd_warshall(graph_c, directed=True, overwrite=True)
    print(f"   Result:\n{result_c}")

    # Test with F-contiguous
    print("\n2. F-contiguous with overwrite=True:")
    graph_f = np.asfortranarray(graph, dtype=np.float64)
    result_f = floyd_warshall(graph_f, directed=True, overwrite=True)
    print(f"   Result:\n{result_f}")

    # Now test with a graph that has actual inf values
    print("\n" + "=" * 60)
    print("GRAPH WITH ACTUAL INF VALUES")
    print("=" * 60)

    graph_inf = np.array([[0, 2, np.inf],
                          [2, 0, 1],
                          [np.inf, 1, 0]], dtype=np.float64)

    print("Input graph with inf:")
    print(graph_inf)

    # Test with C-contiguous
    print("\n1. C-contiguous with overwrite=False:")
    graph_inf_c = np.ascontiguousarray(graph_inf, dtype=np.float64)
    result_inf_c = floyd_warshall(graph_inf_c, directed=True, overwrite=False)
    print(f"   Result:\n{result_inf_c}")

    # Test with F-contiguous
    print("\n2. F-contiguous with overwrite=False:")
    graph_inf_f = np.asfortranarray(graph_inf, dtype=np.float64)
    result_inf_f = floyd_warshall(graph_inf_f, directed=True, overwrite=False)
    print(f"   Result:\n{result_inf_f}")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    test_inf_behavior()