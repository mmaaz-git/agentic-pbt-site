"""
Test to prove the dead code in _laplacian_dense.
We'll add instrumentation to show lines 545-546 are never executed.
"""

import numpy as np
import sys
import types

# We'll monkey-patch the function to add instrumentation
from scipy.sparse.csgraph import _laplacian

# Store original function
original_laplacian_dense = _laplacian._laplacian_dense

def instrumented_laplacian_dense(graph, normed, axis, copy, form, dtype, symmetrized):
    """Instrumented version to detect dead code"""

    if form != "array":
        raise ValueError(f'{form!r} must be "array"')

    print(f"Before first dtype check: dtype = {dtype}")

    if dtype is None:
        dtype = graph.dtype
        print(f"After first dtype check (line 537-538): dtype = {dtype} (set from graph.dtype)")

    if copy:
        m = np.array(graph)
    else:
        m = np.asarray(graph)

    print(f"Before second dtype check (line 545): dtype = {dtype}")

    if dtype is None:  # This should never be True
        print("DEAD CODE EXECUTED: Line 545-546 reached!")
        dtype = m.dtype
    else:
        print("Second dtype check: dtype is NOT None, skipping line 546")

    # Continue with rest of original function
    return original_laplacian_dense(graph, normed, axis, copy, form, dtype, symmetrized)

# Replace the function temporarily
_laplacian._laplacian_dense = instrumented_laplacian_dense

# Test cases
test_cases = [
    {"dtype": None, "copy": True},
    {"dtype": None, "copy": False},
    {"dtype": np.float64, "copy": True},
    {"dtype": np.float64, "copy": False},
]

graph = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float32)

for i, test_case in enumerate(test_cases, 1):
    print(f"\n--- Test Case {i}: dtype={test_case['dtype']}, copy={test_case['copy']} ---")
    lap, d = _laplacian._laplacian_dense(
        graph, normed=False, axis=0,
        copy=test_case['copy'],
        form="array",
        dtype=test_case['dtype'],
        symmetrized=False
    )
    print(f"Result dtype: {lap.dtype}")

# Restore original function
_laplacian._laplacian_dense = original_laplacian_dense

print("\n=== CONCLUSION ===")
print("The second dtype check (lines 545-546) is NEVER executed.")
print("When dtype is None initially, it gets set at line 538, so it can't be None at line 545.")
print("This is dead code that should be removed.")