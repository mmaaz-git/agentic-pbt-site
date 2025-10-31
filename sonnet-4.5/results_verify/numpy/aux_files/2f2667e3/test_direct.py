import numpy.rec
import traceback

print("Testing numpy.rec.array with empty list:")
try:
    result = numpy.rec.array([], dtype=[('value', 'i4')])
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError raised: {e}")
    traceback.print_exc()

print("\n\nTesting numpy.rec.array with empty tuple:")
try:
    result = numpy.rec.array((), dtype=[('value', 'i4')])
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError raised: {e}")
    traceback.print_exc()

print("\n\nTesting regular numpy.array with empty list:")
import numpy as np
try:
    result = np.array([])
    print(f"numpy.array([]) works fine: {result}")
    print(f"Shape: {result.shape}, dtype: {result.dtype}")
except Exception as e:
    print(f"Unexpected error: {e}")