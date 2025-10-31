import numpy.strings as nps
import numpy as np

print("numpy.strings.slice docstring:")
print("=" * 60)
print(nps.slice.__doc__)
print("=" * 60)

print("\nChecking numpy version:")
print(f"NumPy version: {np.__version__}")

# Also check if there's any related documentation in the module
print("\n\nModule docstring:")
print("=" * 60)
print(nps.__doc__)
print("=" * 60)