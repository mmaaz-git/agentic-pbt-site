import scipy.io.matlab as matlab
import inspect

print("=== scipy.io.matlab.savemat documentation ===")
print(matlab.savemat.__doc__)
print("\n" + "="*50 + "\n")

print("=== scipy.io.matlab.loadmat documentation ===")
print(matlab.loadmat.__doc__)
print("\n" + "="*50 + "\n")

# Check for any relevant parameters
sig_savemat = inspect.signature(matlab.savemat)
sig_loadmat = inspect.signature(matlab.loadmat)

print("savemat parameters:", list(sig_savemat.parameters.keys()))
print("loadmat parameters:", list(sig_loadmat.parameters.keys()))