import numpy as np
import scipy.sparse.csgraph as csg

# Create a graph with a very small but non-zero value
graph = np.array([[0.0, 0.0], [0.0, 1.0018225238781444e-157]])

print("="*60)
print("DEMONSTRATING BUG: Silent Data Loss in csgraph_from_dense")
print("="*60)
print()

print("Original graph:")
print(graph)
print()

print(f"Value at position [1,1]: {graph[1,1]}")
print(f"Is this value exactly 0.0? {graph[1,1] == 0.0}")
print(f"Scientific notation: {graph[1,1]:.2e}")
print()

# Convert to sparse and back
print("Converting to sparse representation...")
sparse = csg.csgraph_from_dense(graph, null_value=0)
print(f"Sparse matrix has {sparse.nnz} non-zero elements")
print()

print("Converting back to dense...")
reconstructed = csg.csgraph_to_dense(sparse, null_value=0)
print()

print("Reconstructed graph:")
print(reconstructed)
print()

print(f"Reconstructed value at [1,1]: {reconstructed[1,1]}")
print(f"Is reconstructed value 0.0? {reconstructed[1,1] == 0.0}")
print()

# Verify the bug
print("VERIFICATION:")
print(f"Original value was non-zero: {graph[1,1] != 0.0}")
print(f"Reconstructed value is zero: {reconstructed[1,1] == 0.0}")
print(f"Data was lost: {graph[1,1] != reconstructed[1,1]}")
print()

if graph[1,1] != 0.0 and reconstructed[1,1] == 0.0:
    print("*** BUG CONFIRMED: Non-zero value was silently dropped! ***")
    print()

# Test threshold values
print("="*60)
print("TESTING THRESHOLD BEHAVIOR")
print("="*60)
print()

test_values = [1e-10, 1e-9, 1e-8, 1.1e-8, 1.5e-8, 1e-7, 1e-6]

for val in test_values:
    g = np.array([[0.0, val], [val, 0.0]])
    s = csg.csgraph_from_dense(g, null_value=0)
    r = csg.csgraph_to_dense(s, null_value=0)
    preserved = s.nnz > 0
    status = "PRESERVED" if preserved else "DROPPED"
    print(f"Value {val:.2e}: nnz={s.nnz:2d}, Status={status:9s}, Reconstructed correctly: {np.allclose(g, r, rtol=0, atol=0)}")