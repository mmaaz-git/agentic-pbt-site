from xarray.indexes import RangeIndex

print("Testing direct example from bug report...")
print("Calling RangeIndex.linspace(0.0, 1.0, num=1, endpoint=True, dim='x')")
try:
    index = RangeIndex.linspace(0.0, 1.0, num=1, endpoint=True, dim="x")
    print(f"Success! Result: {index}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError occurred: {e}")
except Exception as e:
    print(f"Other error occurred: {e}")