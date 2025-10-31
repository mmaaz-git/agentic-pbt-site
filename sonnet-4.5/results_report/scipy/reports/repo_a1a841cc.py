import numpy as np
import tempfile
from scipy.io.matlab import savemat

arr = np.array([[1.0]])
mdict = {'aĀ': arr}

with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
    fname = f.name

savemat(fname, mdict, format='5')