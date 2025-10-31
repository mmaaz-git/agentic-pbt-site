import numpy as np
import numpy.ma as ma

empty_arr = ma.array([], dtype=int)
result = ma.flatnotmasked_edges(empty_arr)

print(f"Result: {result}")
print(f"Type of result: {type(result)}")
print(f"Expected: None")
print(f"Actual: [0, -1]")

# Let's verify what the result actually is
if result is not None:
    print(f"Length of result: {len(result)}")
    print(f"First element: {result[0]}")
    print(f"Second element: {result[1]}")
    print(f"Is result[1] < result[0]? {result[1] < result[0]}")
    print(f"Is result[1] negative? {result[1] < 0}")