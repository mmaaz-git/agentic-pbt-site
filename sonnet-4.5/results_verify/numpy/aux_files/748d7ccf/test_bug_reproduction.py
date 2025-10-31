#!/usr/bin/env python3
"""Test case to reproduce the ndpointer empty flag bug"""

import numpy.ctypeslib
import numpy as np
import traceback

print("Testing numpy.ctypeslib.ndpointer with empty flags")
print("="*60)

# Test 1: Empty string flag
print("\nTest 1: ndpointer(flags='')")
try:
    result = numpy.ctypeslib.ndpointer(flags='')
    print(f"Result: {result}")
except KeyError as e:
    print(f"Got KeyError: {e}")
    print(f"Error repr: {repr(e)}")
except TypeError as e:
    print(f"Got TypeError: {e}")
except Exception as e:
    print(f"Got unexpected exception {type(e).__name__}: {e}")

# Test 2: Single comma
print("\nTest 2: ndpointer(flags=',')")
try:
    result = numpy.ctypeslib.ndpointer(flags=',')
    print(f"Result: {result}")
except KeyError as e:
    print(f"Got KeyError: {e}")
    print(f"Error repr: {repr(e)}")
except TypeError as e:
    print(f"Got TypeError: {e}")
except Exception as e:
    print(f"Got unexpected exception {type(e).__name__}: {e}")

# Test 3: Empty element in comma-separated list
print("\nTest 3: ndpointer(flags='C_CONTIGUOUS,,WRITEABLE')")
try:
    result = numpy.ctypeslib.ndpointer(flags='C_CONTIGUOUS,,WRITEABLE')
    print(f"Result: {result}")
except KeyError as e:
    print(f"Got KeyError: {e}")
    print(f"Error repr: {repr(e)}")
except TypeError as e:
    print(f"Got TypeError: {e}")
except Exception as e:
    print(f"Got unexpected exception {type(e).__name__}: {e}")

# Test 4: Trailing comma
print("\nTest 4: ndpointer(flags='C_CONTIGUOUS,')")
try:
    result = numpy.ctypeslib.ndpointer(flags='C_CONTIGUOUS,')
    print(f"Result: {result}")
except KeyError as e:
    print(f"Got KeyError: {e}")
    print(f"Error repr: {repr(e)}")
except TypeError as e:
    print(f"Got TypeError: {e}")
except Exception as e:
    print(f"Got unexpected exception {type(e).__name__}: {e}")

# Test 5: Leading comma
print("\nTest 5: ndpointer(flags=',WRITEABLE')")
try:
    result = numpy.ctypeslib.ndpointer(flags=',WRITEABLE')
    print(f"Result: {result}")
except KeyError as e:
    print(f"Got KeyError: {e}")
    print(f"Error repr: {repr(e)}")
except TypeError as e:
    print(f"Got TypeError: {e}")
except Exception as e:
    print(f"Got unexpected exception {type(e).__name__}: {e}")

# Test 6: List with empty string
print("\nTest 6: ndpointer(flags=[''])")
try:
    result = numpy.ctypeslib.ndpointer(flags=[''])
    print(f"Result: {result}")
except KeyError as e:
    print(f"Got KeyError: {e}")
    print(f"Error repr: {repr(e)}")
except TypeError as e:
    print(f"Got TypeError: {e}")
except Exception as e:
    print(f"Got unexpected exception {type(e).__name__}: {e}")

# Test 7: Valid flag for comparison
print("\nTest 7 (Valid): ndpointer(flags='C_CONTIGUOUS')")
try:
    result = numpy.ctypeslib.ndpointer(flags='C_CONTIGUOUS')
    print(f"Result: {result}")
except Exception as e:
    print(f"Got exception {type(e).__name__}: {e}")

# Test 8: Invalid flag name for comparison
print("\nTest 8 (Invalid name): ndpointer(flags='INVALID_FLAG')")
try:
    result = numpy.ctypeslib.ndpointer(flags='INVALID_FLAG')
    print(f"Result: {result}")
except KeyError as e:
    print(f"Got KeyError: {e}")
    print(f"Error repr: {repr(e)}")
except TypeError as e:
    print(f"Got TypeError: {e}")
except Exception as e:
    print(f"Got unexpected exception {type(e).__name__}: {e}")