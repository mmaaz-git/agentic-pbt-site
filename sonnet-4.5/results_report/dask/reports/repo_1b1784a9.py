import numpy as np
import pandas as pd
import dask.dataframe as dd

df = pd.DataFrame({'x': list(range(100))})
ddf = dd.from_pandas(df, npartitions=4)

q_original = np.array([0.9, 0.5, 0.1])
q_copy = q_original.copy()

print(f"Before: q_original = {q_original}")

result = ddf.x.quantile(q_original)

print(f"After:  q_original = {q_original}")
print(f"Expected:          {q_copy}")

if not np.array_equal(q_original, q_copy):
    print("\n*** BUG: Input array was mutated! ***")
    print(f"Original was: {q_copy}")
    print(f"Now is:       {q_original}")
else:
    print("\n*** No bug detected - array was not mutated ***")