import numpy as np
from scipy import io as scipy_io
import os

print("Testing with filename directly...")

for i in range(5):
    print(f"\n=== Iteration {i+1} ===")
    arr = np.random.rand(i+1, i+1)
    filename = f"test_matrix_{i}.mtx"

    print(f"Writing array of shape {arr.shape} to {filename}")
    scipy_io.mmwrite(filename, arr)

    print(f"Calling mminfo with filename directly")
    info = scipy_io.mminfo(filename)
    print(f"mminfo result: {info}")

    # Cleanup
    os.remove(filename)

print("\nTest completed successfully!")