import numpy as np

# Test how numpy.linspace handles num=1 with different endpoint settings
print("NumPy behavior with num=1:")
print("numpy.linspace(0, 1, 1, endpoint=True):", np.linspace(0, 1, 1, endpoint=True))
print("numpy.linspace(0, 1, 1, endpoint=False):", np.linspace(0, 1, 1, endpoint=False))
print("numpy.linspace(5, 10, 1, endpoint=True):", np.linspace(5, 10, 1, endpoint=True))
print("numpy.linspace(5, 10, 1, endpoint=False):", np.linspace(5, 10, 1, endpoint=False))