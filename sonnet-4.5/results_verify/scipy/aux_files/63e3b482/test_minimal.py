import numpy as np
from io import BytesIO
import scipy.io.matlab as sio
import traceback

data = {'Ā': np.array([[1.0]])}
f = BytesIO()
try:
    sio.savemat(f, data)
    print("Success - no error raised")
except UnicodeEncodeError as e:
    print(f"UnicodeEncodeError: {e}")
    traceback.print_exc()