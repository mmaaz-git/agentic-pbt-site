import numpy as np
from pandas._libs import missing as libmissing
import pandas.core.ops as ops

left_true = np.array([True])
mask_false = np.array([False])

left_na = np.array([False])
mask_true = np.array([True])

result1, mask1 = ops.kleene_xor(left_true, libmissing.NA, mask_false, None)
result2, mask2 = ops.kleene_xor(left_na, True, mask_true, None)

print(f'True ^ NA: result={result1[0]}, mask={mask1[0]}')
print(f'NA ^ True: result={result2[0]}, mask={mask2[0]}')

print(f'\nAssertion check: result1[0] == result2[0]?')
print(f'result1[0] = {result1[0]}, result2[0] = {result2[0]}')
assert result1[0] == result2[0]