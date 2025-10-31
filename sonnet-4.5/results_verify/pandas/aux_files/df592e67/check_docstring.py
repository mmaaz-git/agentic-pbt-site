from pandas.arrays import SparseArray
import inspect

# Get the docstring for SparseArray.astype
print("SparseArray.astype docstring:")
print("="*80)
print(inspect.getdoc(SparseArray.astype))
print("="*80)