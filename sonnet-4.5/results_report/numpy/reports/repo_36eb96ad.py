import warnings
import numpy as np

# Suppress the deprecation warning for cleaner output
with warnings.catch_warnings():
    warnings.simplefilter("ignore", PendingDeprecationWarning)

    # Create a simple matrix
    A = np.matrix([[1, 2]])

    # Try to use bmat with only gdict parameter
    # This should work according to documentation, but crashes
    result = np.bmat('A', gdict={'A': A})
    print("Result:", result)