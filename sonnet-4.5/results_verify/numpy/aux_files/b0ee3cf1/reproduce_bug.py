import numpy as np
import numpy.lib.format as fmt

dtype_with_shape = np.dtype(('i4', (1,)))
print(f"Original dtype: {dtype_with_shape}")
print(f"Original shape: {dtype_with_shape.shape}")

descr = fmt.dtype_to_descr(dtype_with_shape)
print(f"After dtype_to_descr: {descr}")

restored = fmt.descr_to_dtype(descr)
print(f"Restored dtype: {restored}")
print(f"Restored shape: {restored.shape}")

print(f"\nComparison:")
print(f"  dtype_with_shape == restored: {dtype_with_shape == restored}")
print(f"  dtype_with_shape.shape: {dtype_with_shape.shape}")
print(f"  restored.shape: {restored.shape}")

assert restored == dtype_with_shape