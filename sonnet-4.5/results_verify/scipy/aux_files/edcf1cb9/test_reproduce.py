import numpy as np
import scipy.sparse.csgraph as csg

print("Testing the specific example from the bug report:")
print("=" * 60)

graph = np.array([[0.0, 0.0], [0.0, 1.0018225238781444e-157]])

print(f"Original value at [1,1]: {graph[1,1]}")
print(f"Is exactly 0.0? {graph[1,1] == 0.0}")

sparse = csg.csgraph_from_dense(graph, null_value=0)
reconstructed = csg.csgraph_to_dense(sparse, null_value=0)

print(f"Sparse nnz: {sparse.nnz}")
print(f"Reconstructed value at [1,1]: {reconstructed[1,1]}")

assert graph[1,1] != 0.0, "Original value should not be 0.0"
assert reconstructed[1,1] == 0.0, "Reconstructed value is 0.0"
print("BUG CONFIRMED: Non-zero value was silently dropped!")

print("\nTesting threshold behavior:")
print("-" * 40)
for exp in [-10, -9, -8, -7]:
    val = 10.0 ** exp
    g = np.array([[0.0, val], [val, 0.0]])  # Make it square
    s = csg.csgraph_from_dense(g, null_value=0)
    print(f"Value {val:.2e}: nnz={s.nnz} ({'preserved' if s.nnz > 0 else 'DROPPED'})")

print("\nAdditional tests to find exact threshold:")
print("-" * 40)
for exp_val in [5e-9, 7.5e-9, 9e-9, 9.5e-9, 1e-8, 1.5e-8, 2e-8]:
    g = np.array([[0.0, exp_val], [exp_val, 0.0]])  # Make it square
    s = csg.csgraph_from_dense(g, null_value=0)
    print(f"Value {exp_val:.2e}: nnz={s.nnz} ({'preserved' if s.nnz > 0 else 'DROPPED'})")