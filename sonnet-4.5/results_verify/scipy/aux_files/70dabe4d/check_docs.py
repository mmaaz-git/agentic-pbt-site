import scipy.io.matlab as mat
import inspect

# Get the docstrings for loadmat and savemat
print("=" * 80)
print("LOADMAT DOCUMENTATION:")
print("=" * 80)
print(mat.loadmat.__doc__)

print("\n" + "=" * 80)
print("SAVEMAT DOCUMENTATION:")
print("=" * 80)
print(mat.savemat.__doc__)

# Check default values
print("\n" + "=" * 80)
print("LOADMAT SIGNATURE:")
print("=" * 80)
print(inspect.signature(mat.loadmat))

print("\n" + "=" * 80)
print("SAVEMAT SIGNATURE:")
print("=" * 80)
print(inspect.signature(mat.savemat))