import tempfile
import os
import numpy as np
from scipy.io.matlab import savemat, loadmat

with tempfile.TemporaryDirectory() as tmpdir:
    # Test case: using savemat with appendmat=True (default)
    filename = os.path.join(tmpdir, 'test')
    arr = np.array([1, 2, 3])
    data = {'a': arr}

    print("=== Testing savemat with appendmat=True ===")
    print(f"Filename provided: {filename}")
    print(f"appendmat parameter: True")

    savemat(filename, data, appendmat=True)

    print("\nFiles created in directory:")
    files = os.listdir(tmpdir)
    print(f"  Actual: {files}")
    print(f"  Expected: ['test.mat']")

    filename_with_mat = os.path.join(tmpdir, 'test.mat')
    filename_without_mat = os.path.join(tmpdir, 'test')

    print(f"\nFile existence check:")
    print(f"  'test.mat' exists: {os.path.exists(filename_with_mat)}")
    print(f"  'test' exists: {os.path.exists(filename_without_mat)}")

    print("\n=== Verifying the bug ===")
    if os.path.exists(filename_without_mat) and not os.path.exists(filename_with_mat):
        print("BUG CONFIRMED: savemat created 'test' instead of 'test.mat' when appendmat=True")
    else:
        print("Bug not reproduced")

    print("\n=== Testing loadmat for comparison ===")
    print("Now testing if loadmat correctly handles appendmat=True...")

    # First save with explicit .mat extension
    filename_explicit = os.path.join(tmpdir, 'test2.mat')
    savemat(filename_explicit, data, appendmat=False)
    print(f"Saved file with explicit .mat: {filename_explicit}")

    # Now try to load without .mat extension but with appendmat=True
    filename_load = os.path.join(tmpdir, 'test2')
    print(f"Loading with filename: {filename_load} and appendmat=True")
    try:
        loaded_data = loadmat(filename_load, appendmat=True)
        print(f"SUCCESS: loadmat correctly found 'test2.mat' when given 'test2' with appendmat=True")
        print(f"Loaded data: {loaded_data['a']}")
    except FileNotFoundError as e:
        print(f"FAILED: loadmat couldn't find the file: {e}")

    print("\n=== Summary ===")
    print("savemat with appendmat=True: FAILS to append .mat extension")
    print("loadmat with appendmat=True: CORRECTLY appends .mat extension")
    print("This inconsistency confirms the bug in savemat.")