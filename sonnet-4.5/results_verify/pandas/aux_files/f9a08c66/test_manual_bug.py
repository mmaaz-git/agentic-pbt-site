import numpy as np
from pandas.core.dtypes.common import ensure_python_int

float_value = 42.0
result = ensure_python_int(float_value)
print(f"Result: {result}")
print(f"Result type: {type(result)}")
print(f"Input type: {type(float_value)}")
print(f"Input value: {float_value}")

# Test with non-integer float
try:
    result2 = ensure_python_int(42.5)
    print(f"Non-integer float result: {result2}")
except TypeError as e:
    print(f"Non-integer float raised TypeError: {e}")