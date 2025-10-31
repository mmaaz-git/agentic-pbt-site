import numpy as np
import tempfile
import warnings
import os
from scipy.io.matlab import loadmat, savemat, MatWriteWarning

# Create test data
arr = np.array([[1.0]])
mdict = {'1invalid': arr, 'valid': arr}

# Create temporary file
with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
    fname = f.name

try:
    # Save the data
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        savemat(fname, mdict, format='5')

        # Check if MatWriteWarning was issued
        warning_issued = any(issubclass(warn.category, MatWriteWarning) for warn in w)
        print(f"MatWriteWarning issued: {warning_issued}")

        # Show all warnings if any
        if w:
            for warn in w:
                print(f"Warning: {warn.category.__name__}: {warn.message}")
        else:
            print("No warnings were issued")

    # Load the data back
    loaded = loadmat(fname)

    # Check what was loaded
    print(f"\nKeys in loaded file: {list(loaded.keys())}")
    print(f"'1invalid' in loaded file: {'1invalid' in loaded}")
    print(f"'valid' in loaded file: {'valid' in loaded}")

    # If '1invalid' is in the loaded data, print its value
    if '1invalid' in loaded:
        print(f"Value of '1invalid': {loaded['1invalid']}")

finally:
    # Clean up
    if os.path.exists(fname):
        os.unlink(fname)