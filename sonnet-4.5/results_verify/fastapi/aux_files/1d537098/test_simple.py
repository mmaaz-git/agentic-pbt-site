import pandas as pd
import numpy as np
from dask.dataframe.dask_expr import from_pandas

pdf = pd.DataFrame({'x': np.arange(5)}, index=np.arange(5))
df = from_pandas(pdf, npartitions=2)

print(f"Original: npartitions={df.npartitions}, divisions={df.divisions}")

repartitioned = df.repartition(npartitions=4)

print(f"Requested: 4 partitions")
print(f"Got: npartitions={repartitioned.npartitions}, divisions={repartitioned.divisions}")
print(f"Division count: {len(repartitioned.divisions)} (expected 5)")
print(f"Actual npartitions: {repartitioned.npartitions} (expected 4)")

# Verify the invariant
print(f"\nInvariant check:")
print(f"len(divisions) - 1 = {len(repartitioned.divisions) - 1}")
print(f"npartitions = {repartitioned.npartitions}")
print(f"Are they equal? {len(repartitioned.divisions) - 1 == repartitioned.npartitions}")