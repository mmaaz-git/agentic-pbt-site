import numpy as np
import numpy.ma as ma

# This should work but crashes with AttributeError
result = ma.default_fill_value(np.float32)
print(f"Result: {result}")