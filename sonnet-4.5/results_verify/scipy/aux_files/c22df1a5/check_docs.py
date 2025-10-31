import scipy.sparse.linalg as sla
import inspect

# Get the docstring for expm
print("Full docstring for scipy.sparse.linalg.expm:")
print("=" * 80)
print(sla.expm.__doc__)
print("=" * 80)

# Get source location
print(f"\nSource file: {inspect.getfile(sla.expm)}")