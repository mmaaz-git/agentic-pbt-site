import tempfile
import os
from scipy.io.matlab import savemat, loadmat

with tempfile.TemporaryDirectory() as tmpdir:
    fname = os.path.join(tmpdir, 'test')
    data = {'x': 1.0}

    print("Testing savemat with appendmat=True:")
    savemat(fname, data, appendmat=True)

    print(f"Files in directory: {os.listdir(tmpdir)}")
    print(f"Expected 'test.mat': {os.path.exists(fname + '.mat')}")
    print(f"Actual 'test' exists: {os.path.exists(fname)}")

    # Try to load the file with appendmat=True to see if loadmat works differently
    print("\nTesting loadmat with appendmat=True:")
    try:
        loaded = loadmat(fname, appendmat=True)
        print("Loaded successfully from 'test' with appendmat=True")
        print(f"Data: {loaded['x']}")
    except Exception as e:
        print(f"Failed to load: {e}")

    # Also test with appendmat=False
    print("\nTesting savemat with appendmat=False:")
    fname2 = os.path.join(tmpdir, 'test2')
    savemat(fname2, data, appendmat=False)
    print(f"Files in directory after second save: {os.listdir(tmpdir)}")
    print(f"'test2' exists: {os.path.exists(fname2)}")
    print(f"'test2.mat' exists: {os.path.exists(fname2 + '.mat')}")