import tempfile
import os
import numpy as np
from scipy.io.matlab import savemat, loadmat

print("Testing scipy.io.matlab.savemat appendmat parameter...")
print("=" * 60)

with tempfile.TemporaryDirectory() as tmpdir:
    filename = os.path.join(tmpdir, 'test')
    arr = np.array([1, 2, 3])
    data = {'a': arr}

    print(f"Calling savemat('{filename}', data, appendmat=True)")
    savemat(filename, data, appendmat=True)

    print("\nFiles created:", os.listdir(tmpdir))
    print("Expected: ['test.mat']")
    print("Actual:", os.listdir(tmpdir))

    filename_with_mat = os.path.join(tmpdir, 'test.mat')
    print(f"\ntest.mat exists: {os.path.exists(filename_with_mat)}")

    filename_without_mat = os.path.join(tmpdir, 'test')
    print(f"test exists: {os.path.exists(filename_without_mat)}")

print("\n" + "=" * 60)
print("Testing loadmat with appendmat=True...")

with tempfile.TemporaryDirectory() as tmpdir:
    # Write with explicit .mat extension
    filename_write = os.path.join(tmpdir, 'test.mat')
    savemat(filename_write, data, appendmat=False)
    print(f"Written file with explicit .mat: {os.listdir(tmpdir)}")

    # Read without .mat extension but with appendmat=True
    filename_read = os.path.join(tmpdir, 'test')
    loaded = loadmat(filename_read, appendmat=True)
    print(f"Successfully loaded using appendmat=True: {list(loaded.keys())}")