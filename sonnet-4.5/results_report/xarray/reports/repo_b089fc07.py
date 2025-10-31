import numpy as np
from xarray.plot.utils import _infer_interval_breaks

# Test case with unsorted coordinates
coord = np.array([0.0, -1.0])
result = _infer_interval_breaks(coord)

print(f"Input coordinates: {coord}")
print(f"Data range: [{coord.min()}, {coord.max()}]")
print(f"Interval breaks: {result}")
print(f"Breaks range: [{result[0]}, {result[-1]}]")
print()
print("Analysis:")
print(f"- First break ({result[0]}) should be <= minimum value ({coord.min()})")
print(f"  Result: {result[0]} <= {coord.min()} is {result[0] <= coord.min()}")
print(f"- Last break ({result[-1]}) should be >= maximum value ({coord.max()})")
print(f"  Result: {result[-1]} >= {coord.max()} is {result[-1] >= coord.max()}")
print()

# Additional test with descending sorted coordinates
coord_desc = np.array([1.0, 0.0, -1.0])
result_desc = _infer_interval_breaks(coord_desc)

print("Test with descending sorted coordinates:")
print(f"Input coordinates: {coord_desc}")
print(f"Data range: [{coord_desc.min()}, {coord_desc.max()}]")
print(f"Interval breaks: {result_desc}")
print(f"Breaks range: [{result_desc[0]}, {result_desc[-1]}]")
print(f"- First break ({result_desc[0]}) <= minimum ({coord_desc.min()}): {result_desc[0] <= coord_desc.min()}")
print(f"- Last break ({result_desc[-1]}) >= maximum ({coord_desc.max()}): {result_desc[-1] >= coord_desc.max()}")
print()

# Test with ascending sorted coordinates (should work correctly)
coord_asc = np.array([-1.0, 0.0, 1.0])
result_asc = _infer_interval_breaks(coord_asc)

print("Test with ascending sorted coordinates (control case):")
print(f"Input coordinates: {coord_asc}")
print(f"Data range: [{coord_asc.min()}, {coord_asc.max()}]")
print(f"Interval breaks: {result_asc}")
print(f"Breaks range: [{result_asc[0]}, {result_asc[-1]}]")
print(f"- First break ({result_asc[0]}) <= minimum ({coord_asc.min()}): {result_asc[0] <= coord_asc.min()}")
print(f"- Last break ({result_asc[-1]}) >= maximum ({coord_asc.max()}): {result_asc[-1] >= coord_asc.max()}")