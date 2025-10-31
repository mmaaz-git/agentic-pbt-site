import numpy as np
import numpy.ma as ma

# Create an empty masked array
empty_arr = ma.array([], dtype=int)

# Call flatnotmasked_edges on the empty array
result = ma.flatnotmasked_edges(empty_arr)

# Print the results
print(f"Input array: {empty_arr}")
print(f"Array shape: {empty_arr.shape}")
print(f"Array size: {empty_arr.size}")
print(f"Result type: {type(result)}")
print(f"Result: {result}")

if result is not None:
    print(f"First index: {result[0]}")
    print(f"Second index: {result[1]}")
    print(f"Is second index >= first index? {result[1] >= result[0]}")
    print(f"Are indices valid for empty array? No - array has size 0")