import pandas as pd
import dask.dataframe as dd

# Monkey-patch to add debug info
import dask.dataframe.dask_expr._repartition as rep_module

original_partitions_boundaries = rep_module.RepartitionToFewer._partitions_boundaries

def debug_partitions_boundaries(self):
    npartitions = self.new_partitions
    npartitions_input = self.frame.npartitions
    print(f"DEBUG _partitions_boundaries:")
    print(f"  npartitions_input (frame.npartitions) = {npartitions_input}")
    print(f"  npartitions (new_partitions) = {npartitions}")
    print(f"  frame type = {type(self.frame)}")
    if hasattr(self.frame, '_name'):
        print(f"  frame._name = {self.frame._name}")

    # The assertion that fails
    if not (npartitions_input > npartitions):
        print(f"  ASSERTION WILL FAIL: {npartitions_input} > {npartitions} is FALSE")
        # Let's see what the frame actually is
        print(f"  Frame details:")
        print(f"    frame = {self.frame}")
        if hasattr(self.frame, 'operands'):
            print(f"    frame.operands = {self.frame.operands}")

    return original_partitions_boundaries.fget(self)

# Replace the property
rep_module.RepartitionToFewer._partitions_boundaries = property(debug_partitions_boundaries)

# Now run the test
series = pd.Series(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    index=pd.date_range('2000-01-01 00:00:00', periods=10, freq='h')
)

npartitions = 5
ds = dd.from_pandas(series, npartitions=npartitions)

print("Starting resample operation...")
try:
    result = ds.resample('D').count().compute()
    print(f"Result: {result}")
except AssertionError as e:
    print(f"AssertionError as expected")