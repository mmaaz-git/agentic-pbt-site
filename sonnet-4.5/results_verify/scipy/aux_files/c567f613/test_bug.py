import numpy as np
import tempfile
import os
import warnings
from scipy.io.matlab import loadmat, savemat, MatWriteWarning

arr = np.array([1, 2, 3])
varname = '0test'

with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
    fname = f.name

try:
    mdict = {varname: arr}

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        savemat(fname, mdict)
        print(f"Warnings raised: {len(w)}")
        if len(w) > 0:
            for warning in w:
                print(f"Warning: {warning.message}")

    result = loadmat(fname)

    if varname in result:
        print(f"BUG: Variable '{varname}' was saved despite starting with a digit!")
        print(f"Loaded value: {result[varname]}")
    else:
        print(f"Variable '{varname}' was correctly not saved")
        print(f"Keys in result: {[k for k in result.keys() if not k.startswith('__')]}")
finally:
    if os.path.exists(fname):
        os.unlink(fname)