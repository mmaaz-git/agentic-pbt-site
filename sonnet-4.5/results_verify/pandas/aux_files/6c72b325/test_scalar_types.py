import pandas as pd
import numpy as np
from pandas.api.types import is_scalar

# Test what pandas considers to be scalar
test_values = [
    (0, "Python int"),
    (1.5, "Python float"),
    (True, "Python bool"),
    (1+2j, "Python complex"),
    (None, "None"),
    ("hello", "string"),
    (b"bytes", "bytes"),
    (np.int64(5), "numpy int64"),
    (np.float64(5.5), "numpy float64"),
    ([1, 2, 3], "list"),
    ((1, 2), "tuple"),
    ({1, 2}, "set"),
    ({"a": 1}, "dict"),
]

print("Testing what pandas considers to be scalar using is_scalar():\n")
for val, desc in test_values:
    try:
        result = is_scalar(val)
        print(f"{desc:20} ({val}): is_scalar = {result}")
    except Exception as e:
        print(f"{desc:20} ({val}): ERROR - {e}")