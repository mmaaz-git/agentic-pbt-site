import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

from xarray.indexes import RangeIndex

print("Testing RangeIndex.linspace with num=1, endpoint=True")
try:
    index = RangeIndex.linspace(0.0, 1.0, num=1, endpoint=True, dim="x")
    print(f"Success: Created index with size={index.size}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {e}")

print("\nTesting RangeIndex.linspace with num=1, endpoint=False")
try:
    index = RangeIndex.linspace(0.0, 1.0, num=1, endpoint=False, dim="x")
    print(f"Success: Created index with size={index.size}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {e}")

print("\nTesting with num=2, endpoint=True as control")
try:
    index = RangeIndex.linspace(0.0, 1.0, num=2, endpoint=True, dim="x")
    print(f"Success: Created index with size={index.size}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {e}")