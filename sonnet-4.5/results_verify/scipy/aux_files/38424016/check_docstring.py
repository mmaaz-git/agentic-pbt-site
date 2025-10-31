import scipy.sparse.linalg as spl
import inspect

print("=== scipy.sparse.linalg.expm docstring ===")
print(spl.expm.__doc__)

print("\n=== Source file location ===")
print(inspect.getfile(spl.expm))

print("\n=== Source code snippet ===")
try:
    source = inspect.getsource(spl.expm)
    lines = source.split('\n')
    # Print first 50 lines or so to see the docstring
    for i, line in enumerate(lines[:80]):
        print(f"{i+1:3}: {line}")
except:
    print("Could not get source code")