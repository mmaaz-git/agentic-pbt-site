import numpy as np
from pandas._libs.internals import BlockPlacement
from pandas.core.internals.api import maybe_infer_ndim, make_block
from pandas.core.internals.managers import BlockManager
from pandas import Index

placement = BlockPlacement(slice(0, 1))

# Case 1: 0D array (scalar)
print("=== Case 1: 0D array (scalar) ===")
scalar = np.array(5)
result = maybe_infer_ndim(scalar, placement, ndim=None)
print(f"0D array: maybe_infer_ndim returned {result}, expected 1 or 2")
print(f"Array shape: {scalar.shape}, ndim: {scalar.ndim}")
print()

# Case 2: 3D array
print("=== Case 2: 3D array ===")
array_3d = np.array([[[1, 2], [3, 4]]])
result_3d = maybe_infer_ndim(array_3d, placement, ndim=None)
print(f"3D array: maybe_infer_ndim returned {result_3d}, expected 1 or 2")
print(f"Array shape: {array_3d.shape}, ndim: {array_3d.ndim}")
print()

# Case 3: 4D array
print("=== Case 3: 4D array ===")
array_4d = np.array([[[[1, 2]]]])
result_4d = maybe_infer_ndim(array_4d, placement, ndim=None)
print(f"4D array: maybe_infer_ndim returned {result_4d}, expected 1 or 2")
print(f"Array shape: {array_4d.shape}, ndim: {array_4d.ndim}")
print()

# Downstream impact: Block creation succeeds but BlockManager creation fails
print("=== Downstream Impact ===")
print("Creating Block with 3D array...")
block_3d = make_block(array_3d, placement, ndim=None)
print(f"Block created successfully with ndim={block_3d.ndim}")
print()

print("Attempting to create BlockManager with 3D block...")
try:
    mgr = BlockManager((block_3d,), [Index([0]), Index([0, 1])])
    print("BlockManager created successfully (unexpected!)")
except AssertionError as e:
    print(f"BlockManager creation failed with AssertionError:")
    print(f"  {e}")