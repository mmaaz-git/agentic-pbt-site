#!/usr/bin/env python3
"""Test how the divisions are used in ResampleAggregation"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

import pandas as pd
import numpy as np

# Create a simple dataframe to test resampling
dates = pd.date_range('2000-01-01', periods=25, freq='h')
df = pd.DataFrame({'values': np.arange(25)}, index=dates)

print("Testing pandas resample behavior:")
print("="*60)

# Test with pandas directly
result = df.resample('D', closed='right', label='right').sum()
print(f"Pandas resample result with closed='right', label='right':")
print(result)
print(f"Result index: {result.index.tolist()}")
print(f"Result values: {result.values.flatten().tolist()}")

# Now let's see what dask expects
print("\nAnalyzing ResampleAggregation._divisions() usage:")
print("="*60)

class BlockwiseDep:
    def __init__(self, iterable):
        self.iterable = iterable

# Simulate what ResampleAggregation._divisions() does
divisions_left = BlockwiseDep([pd.Timestamp('2000-01-01 00:00:00'),
                                pd.Timestamp('2000-01-01 00:00:00.000000001')])
divisions_right = BlockwiseDep([pd.Timestamp('2000-01-02 00:00:00')])

# This is what line 193-194 does
result_divisions = list(divisions_left.iterable) + [divisions_right.iterable[-1]]
print(f"Expected divisions format: {result_divisions}")

# But with mismatched lengths:
divisions_left_bad = BlockwiseDep([pd.Timestamp('2000-01-01 00:00:00'),
                                    pd.Timestamp('2000-01-01 00:00:00.000000001'),
                                    pd.Timestamp('2000-01-02 00:00:00.000000001')])
divisions_right_bad = BlockwiseDep([pd.Timestamp('2000-01-01 00:00:00'),
                                     pd.Timestamp('2000-01-02 00:00:00')])

# This would be problematic
print(f"\nWith mismatched lengths:")
print(f"divisions_left has {len(divisions_left_bad.iterable)} elements")
print(f"divisions_right has {len(divisions_right_bad.iterable)} elements")

# When used in ResampleAggregation._lower() line 162-164:
# The BlockwiseDep objects are created like this:
# BlockwiseDep(output_divisions[:-1])  <- for divisions_left
# BlockwiseDep(output_divisions[1:])   <- for divisions_right

print(f"\nIn ResampleAggregation._lower():")
bin_divs = (pd.Timestamp('2000-01-01 00:00:00'),
            pd.Timestamp('2000-01-01 00:00:00.000000001'),
            pd.Timestamp('2000-01-02 00:00:00.000000001'))
out_divs = (pd.Timestamp('2000-01-01 00:00:00'),
            pd.Timestamp('2000-01-02 00:00:00'))

print(f"bin_divs: {bin_divs}")
print(f"out_divs: {out_divs}")
print(f"BlockwiseDep(out_divs[:-1]) would be: {out_divs[:-1]}")
print(f"BlockwiseDep(out_divs[1:]) would be: {out_divs[1:]}")
print(f"Length of out_divs[:-1]: {len(out_divs[:-1])}")
print(f"Length of out_divs[1:]: {len(out_divs[1:])}")

print("\nThis creates inconsistent partition boundaries!")