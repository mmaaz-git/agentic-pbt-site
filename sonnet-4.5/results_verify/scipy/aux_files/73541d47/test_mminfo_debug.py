import tempfile
import numpy as np
from scipy import io as scipy_io
import sys

print("Starting mminfo repeated calls test with debug output...")
sys.stdout.flush()

for i in range(5):
    print(f"\n=== Iteration {i+1} ===")
    sys.stdout.flush()

    arr = np.random.rand(i+1, i+1)
    print(f"Created array of shape {arr.shape}")
    sys.stdout.flush()

    with tempfile.NamedTemporaryFile(mode='w+b', suffix='.mtx', delete=False) as f:
        filename = f.name
        print(f"Writing to file: {filename}")
        sys.stdout.flush()

        scipy_io.mmwrite(f, arr)
        f.flush()
        f.seek(0)

        print("Calling mminfo...")
        sys.stdout.flush()

        info = scipy_io.mminfo(f)
        print(f"mminfo result: {info}")
        sys.stdout.flush()

print("\nTest completed successfully!")
sys.stdout.flush()