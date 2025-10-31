import numpy as np
from scipy.odr import Data, ODR, unilinear
import tempfile

# Create minimal ODR setup
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
data = Data(x, y)
odr_obj = ODR(data, unilinear, beta0=[1.0, 0.0])

# Set up a temp file for reporting
with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
    odr_obj.rptfile = f.name

# This should fail with a confusing error message
# The init parameter should only accept 0, 1, or 2
# but passing 3 gives a cryptic "is not in list" error
odr_obj.set_iprint(init=3)