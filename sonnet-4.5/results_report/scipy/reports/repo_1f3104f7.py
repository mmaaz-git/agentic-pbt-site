import numpy as np
from scipy.sparse.csgraph import csgraph_from_masked

# Create a simple 2x2 matrix with all non-zero values
graph = np.array([[1., 1.], [1., 1.]])

# Create a masked array where we mask values equal to 0
# Since there are no zeros, nothing gets masked
masked_graph = np.ma.masked_equal(graph, 0)

# Print information about the mask
print(f"Graph array:\n{graph}")
print(f"\nMasked graph data:\n{masked_graph.data}")
print(f"\nMask: {masked_graph.mask}")
print(f"Mask type: {type(masked_graph.mask)}")
print(f"Mask shape: {masked_graph.mask.shape if hasattr(masked_graph.mask, 'shape') else 'No shape attribute'}")
print(f"Mask ndim: {masked_graph.mask.ndim if hasattr(masked_graph.mask, 'ndim') else 'No ndim attribute'}")

# Try to convert to sparse graph - this should crash
print("\nAttempting to convert to sparse graph...")
try:
    sparse_graph = csgraph_from_masked(masked_graph)
    print("Success! Sparse graph created.")
    print(f"Sparse graph:\n{sparse_graph}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()