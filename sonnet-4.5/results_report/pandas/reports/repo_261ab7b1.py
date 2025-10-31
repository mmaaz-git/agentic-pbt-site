import numpy as np
import pandas as pd
from pandas.core.window.common import prep_binary

s1 = pd.Series([1.0, 2.0, 3.0])
s2 = pd.Series([10.0, np.inf, 30.0])

X, Y = prep_binary(s1, s2)

print(f"s1: {s1.values}")
print(f"s2: {s2.values}")
print(f"X: {X.values}")
print(f"Y: {Y.values}")

assert not np.isnan(s1.iloc[1]), "s1[1] is finite"
assert np.isinf(s2.iloc[1]), "s2[1] is infinity"
assert np.isnan(X.iloc[1]), "BUG: X[1] became NaN!"