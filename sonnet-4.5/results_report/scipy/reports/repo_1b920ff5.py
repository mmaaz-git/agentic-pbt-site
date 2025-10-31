import tempfile
import os
from scipy.io.matlab import savemat

with tempfile.TemporaryDirectory() as tmpdir:
    fname = os.path.join(tmpdir, 'test')
    data = {'x': 1.0}

    # Test with appendmat=True (should append .mat extension)
    savemat(fname, data, appendmat=True)

    print(f"Files in directory: {os.listdir(tmpdir)}")
    print(f"Expected 'test.mat' exists: {os.path.exists(fname + '.mat')}")
    print(f"Actual 'test' exists: {os.path.exists(fname)}")

    # Also test with appendmat=False for comparison
    fname2 = os.path.join(tmpdir, 'test2')
    savemat(fname2, data, appendmat=False)
    print(f"\nWith appendmat=False:")
    print(f"Files in directory: {os.listdir(tmpdir)}")
    print(f"'test2' exists: {os.path.exists(fname2)}")
    print(f"'test2.mat' exists: {os.path.exists(fname2 + '.mat')}")