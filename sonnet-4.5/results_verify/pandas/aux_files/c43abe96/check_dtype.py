import pandas.arrays as arrays
import numpy as np
import pandas as pd

# Check the docstring
print("NumpyExtensionArray docstring:")
print(arrays.NumpyExtensionArray.__doc__)
print("\n" + "="*50 + "\n")

# Check dtype property docstring
print("dtype property docstring:")
if hasattr(arrays.NumpyExtensionArray, 'dtype'):
    print(arrays.NumpyExtensionArray.dtype.__doc__)
print("\n" + "="*50 + "\n")

# Look at NumpyEADtype class
from pandas.core.dtypes.dtypes import NumpyEADtype
print("NumpyEADtype docstring:")
print(NumpyEADtype.__doc__)