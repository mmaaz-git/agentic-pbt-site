import numpy as np
import tempfile
import warnings
import os
from scipy.io.matlab import loadmat, savemat, MatWriteWarning

arr = np.array([[1.0]])
mdict = {'_invalid': arr, 'valid': arr}

with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
    fname = f.name

try:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        savemat(fname, mdict, format='5')

        print(f"MatWriteWarning issued: {any(issubclass(warn.category, MatWriteWarning) for warn in w)}")
        if w:
            for warning in w:
                print(f"Warning: {warning.message}")

    loaded = loadmat(fname)
    print(f"'_invalid' in loaded file: {'_invalid' in loaded}")
    print(f"'valid' in loaded file: {'valid' in loaded}")
    print(f"Keys in loaded file: {list(loaded.keys())}")
finally:
    if os.path.exists(fname):
        os.unlink(fname)