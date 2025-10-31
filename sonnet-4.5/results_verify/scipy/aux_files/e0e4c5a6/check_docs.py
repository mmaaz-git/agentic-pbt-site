import scipy.ndimage
import numpy as np

# Check the docstring
print("SCIPY NDIMAGE ROTATE DOCSTRING:")
print("="*50)
print(scipy.ndimage.rotate.__doc__)

# Also check the actual implementation to understand reshape=False behavior
help(scipy.ndimage.rotate)