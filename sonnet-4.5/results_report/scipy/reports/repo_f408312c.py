import numpy as np
import scipy.sparse as sp

A = sp.dia_array(np.zeros((1, 1)))
B = sp.dia_array(np.zeros((1, 1)))

result = A @ B