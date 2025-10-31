import tempfile
import numpy as np
from scipy import io as scipy_io

print("Starting mminfo repeated calls test...")

for i in range(5):
    print(f"Iteration {i+1}...")
    arr = np.random.rand(i+1, i+1)
    print(f"Created array of shape {arr.shape}")

    with tempfile.NamedTemporaryFile(mode='w+b', suffix='.mtx', delete=False) as f:
        filename = f.name
        scipy_io.mmwrite(f, arr)
        f.flush()
        f.seek(0)
        info = scipy_io.mminfo(f)
        print(f"mminfo result: {info}")

print("Test completed successfully!")