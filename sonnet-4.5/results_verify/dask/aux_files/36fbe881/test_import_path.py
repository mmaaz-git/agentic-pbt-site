import sys

# Test different import paths
from dask.utils import format_bytes as utils_format_bytes
from dask.widgets.widgets import format_bytes as widgets_format_bytes

print("Testing import paths:")
print(f"dask.utils.format_bytes is {utils_format_bytes}")
print(f"dask.widgets.widgets.format_bytes is {widgets_format_bytes}")
print(f"Are they the same? {utils_format_bytes is widgets_format_bytes}")

# Test that they produce same output
n = 1_125_894_277_343_089_729
result1 = utils_format_bytes(n)
result2 = widgets_format_bytes(n)
print(f"\nFor n = {n}:")
print(f"utils.format_bytes(n) = {result1!r}")
print(f"widgets.format_bytes(n) = {result2!r}")
print(f"Same result? {result1 == result2}")