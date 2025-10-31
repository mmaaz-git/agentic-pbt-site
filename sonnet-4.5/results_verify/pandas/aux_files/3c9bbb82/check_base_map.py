#!/usr/bin/env python3
import pandas
from pandas.core.arrays.base import ExtensionArray
import inspect

print('ExtensionArray has map method:', hasattr(ExtensionArray, 'map'))

if hasattr(ExtensionArray, 'map'):
    print("\nExtensionArray.map source (first 1000 chars):")
    print(inspect.getsource(ExtensionArray.map)[:1000])

# Check the MRO
from pandas.arrays import SparseArray
print("\nSparseArray MRO:")
for cls in SparseArray.__mro__:
    print(f"  {cls}")
    if hasattr(cls, 'map') and cls != SparseArray:
        print(f"    -> has map method")