import numpy as np
from scipy.spatial.distance import is_valid_dm

# Create a 2x2 matrix with non-zero diagonal
mat = np.array([[5.0, 1.0], [1.0, 5.0]])

# This should raise ValueError but raises TypeError instead
# when name=None and tol > 0
is_valid_dm(mat, tol=0.1, throw=True, name=None)