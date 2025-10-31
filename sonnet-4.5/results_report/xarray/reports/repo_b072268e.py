from xarray.backends.netcdf3 import is_valid_nc3_name

# Test empty string which should return False but crashes instead
result = is_valid_nc3_name("")
print(f"Result: {result}")