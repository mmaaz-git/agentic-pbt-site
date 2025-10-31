from scipy.io.matlab import savemat, loadmat
import inspect

print("=== SAVEMAT DOCUMENTATION ===")
print(savemat.__doc__)
print("\n" + "="*50 + "\n")

print("=== LOADMAT DOCUMENTATION ===")
print(loadmat.__doc__)
print("\n" + "="*50 + "\n")

# Get the signature
print("=== SAVEMAT SIGNATURE ===")
print(inspect.signature(savemat))