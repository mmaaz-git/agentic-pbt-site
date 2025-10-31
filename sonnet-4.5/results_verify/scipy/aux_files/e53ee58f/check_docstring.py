from scipy.io import savemat
import inspect

# Get the docstring
docstring = inspect.getdoc(savemat)
print("scipy.io.savemat docstring:")
print("=" * 80)
print(docstring)
print("=" * 80)