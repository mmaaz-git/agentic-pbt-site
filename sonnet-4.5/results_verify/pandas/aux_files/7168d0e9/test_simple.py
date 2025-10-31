import numpy as np
from pandas.core.dtypes.common import is_numeric_v_string_like

arr = np.array([0])
s = '0'

result1 = is_numeric_v_string_like(arr, s)
result2 = is_numeric_v_string_like(s, arr)

print(f"is_numeric_v_string_like(arr, s) = {result1}")
print(f"is_numeric_v_string_like(s, arr) = {result2}")
print(f"Are they equal? {result1 == result2}")