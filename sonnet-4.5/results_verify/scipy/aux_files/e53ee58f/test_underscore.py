import io
import warnings
from scipy.io import savemat, loadmat
from scipy.io.matlab import MatWriteWarning

file_obj = io.BytesIO()
data = {'_test': 123, 'validname': 456}

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    savemat(file_obj, data)

    mat_warnings = [warning for warning in w if issubclass(warning.category, MatWriteWarning)]
    print(f"MatWriteWarnings issued: {len(mat_warnings)}")
    if mat_warnings:
        for warning in mat_warnings:
            print(f"Warning message: {warning.message}")

file_obj.seek(0)
loaded = loadmat(file_obj)
user_keys = [k for k in loaded.keys() if not k.startswith('__')]
print(f"Keys in saved file: {user_keys}")