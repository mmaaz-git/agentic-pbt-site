import numpy as np
import numpy.ma as ma

input_array = np.array([False])
result = ma.make_mask(input_array)

print(f"Input shape: {input_array.shape}")
print(f"Result shape: {result.shape if hasattr(result, 'shape') else 'scalar'}")
print(f"Result: {result}")
print(f"Type of result: {type(result)}")

print("\nComparison with [True]:")
result_true = ma.make_mask(np.array([True]))
print(f"make_mask([True]) shape: {result_true.shape}")
print(f"make_mask([False]) shape: {result.shape if hasattr(result, 'shape') else 'scalar'}")

print("\nComparison with [False, True]:")
result_mixed = ma.make_mask(np.array([False, True]))
print(f"make_mask([False, True]) shape: {result_mixed.shape}")

print("\nAdditional tests:")
# Test with explicit shrink parameter
result_no_shrink = ma.make_mask(np.array([False]), shrink=False)
print(f"make_mask([False], shrink=False) shape: {result_no_shrink.shape}")

result_shrink = ma.make_mask(np.array([False]), shrink=True)
print(f"make_mask([False], shrink=True) shape: {result_shrink.shape if hasattr(result_shrink, 'shape') else 'scalar'}")
print(f"make_mask([False], shrink=True) value: {result_shrink}")