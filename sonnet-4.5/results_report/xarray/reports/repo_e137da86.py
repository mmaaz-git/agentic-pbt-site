from xarray.backends.netcdf3 import is_valid_nc3_name

# Test with empty string to reproduce the bug
result = is_valid_nc3_name("")
print(f"Result: {result}")