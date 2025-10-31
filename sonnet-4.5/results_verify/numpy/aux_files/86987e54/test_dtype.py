import numpy as np
import traceback

# Test how numpy.dtype handles non-string field names
try:
    dt = np.dtype([('f0', 'i4'), (1, 'i4')])  # Second field name is integer
    print(f"Created dtype: {dt}")
except (TypeError, ValueError) as e:
    print(f"numpy.dtype raised {type(e).__name__}: {e}")
except Exception as e:
    print(f"numpy.dtype raised unexpected {type(e).__name__}: {e}")
    traceback.print_exc()

# Also test with list of field names
try:
    dt = np.dtype({'names': [0, 1], 'formats': ['i4', 'i4']})
    print(f"Created dtype with integer names: {dt}")
except (TypeError, ValueError) as e:
    print(f"\nnumpy.dtype with dict raised {type(e).__name__}: {e}")
except Exception as e:
    print(f"\nnumpy.dtype with dict raised unexpected {type(e).__name__}: {e}")
    traceback.print_exc()