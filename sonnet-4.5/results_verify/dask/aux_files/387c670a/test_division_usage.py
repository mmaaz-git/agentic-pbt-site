#!/usr/bin/env python3
"""Test how the divisions are actually used in the ResampleReduction class"""

import pandas as pd
from dask.dataframe.tseries.resample import _resample_bin_and_out_divs

# Create test case
start = pd.Timestamp('2000-01-01')
end = start + pd.Timedelta(days=30)
divisions = pd.date_range(start, end, periods=12)

# Get the problematic divisions
newdivs, outdivs = _resample_bin_and_out_divs(divisions, '3D', closed='right', label='right')

print("Division Analysis")
print("=" * 60)
print(f"newdivs length: {len(newdivs)}")
print(f"outdivs length: {len(outdivs)}")
print(f"Mismatch: {len(newdivs) != len(outdivs)}")

# Show how it's used in _lower method (lines 162-163)
print("\nHow divisions are used in ResampleReduction._lower():")
print("  Line 162: BlockwiseDep(output_divisions[:-1])")
print("  Line 163: BlockwiseDep(output_divisions[1:])")
print("  Line 164: BlockwiseDep(['left'] * (len(output_divisions[1:]) - 1) + [None])")

print("\nWith our outdivs:")
print(f"  outdivs[:-1] has length: {len(outdivs[:-1])} elements")
print(f"  outdivs[1:] has length: {len(outdivs[1:])} elements")
print(f"  'left' list length: {len(outdivs[1:]) - 1} + 1 = {len(outdivs[1:])}")

print("\nAnd in _divisions() method (line 194):")
print("  return list(self.divisions_left.iterable) + [self.divisions_right.iterable[-1]]")
print(f"  Would create: list of {len(outdivs[:-1])} + 1 = {len(outdivs[:-1]) + 1} elements")

# Check what the BlockwiseDep structure looks like
from dask.dataframe.tseries.resample import BlockwiseDep

left = BlockwiseDep(outdivs[:-1])
right = BlockwiseDep(outdivs[1:])
closed = BlockwiseDep(['left'] * (len(outdivs[1:]) - 1) + [None])

print("\nBlockwiseDep structures created:")
print(f"  divisions_left: {len(left.iterable)} elements")
print(f"  divisions_right: {len(right.iterable)} elements")
print(f"  closed: {len(closed.iterable)} elements")

# They should all have the same length for blockwise operations
if len(left.iterable) == len(right.iterable) == len(closed.iterable):
    print("\n✓ All BlockwiseDep have same length - will work correctly")
else:
    print("\n✗ BlockwiseDep length mismatch - could cause problems!")

# Now check what happens with newdivs
print("\n" + "=" * 60)
print("What about newdivs?")
print(f"newdivs is used for Repartition at line 157:")
print(f"  Repartition(frame, new_divisions=newdivs, force=True)")
print(f"  newdivs has {len(newdivs)} elements")
print(f"  Original frame divisions has {len(divisions)} elements")

if len(newdivs) == len(divisions):
    print("✓ newdivs matches original divisions length")
else:
    print("✗ newdivs does NOT match original divisions length")