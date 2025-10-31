import pyarrow as pa
import pyarrow.compute as pc

# Test PyArrow's list_slice directly
lists = [[0, 1, 2, 3], [4, 5, 6]]
pa_array = pa.array(lists, type=pa.list_(pa.int64()))

print("Testing PyArrow's list_slice behavior:")
print(f"Input array: {pa_array.to_pylist()}")

# Test various slices
test_cases = [
    (0, 0),   # Empty slice at start
    (1, 1),   # Empty slice in middle
    (2, 2),   # Another empty slice
    (0, 2),   # Normal slice
    (1, 3),   # Another normal slice
]

for start, stop in test_cases:
    try:
        result = pc.list_slice(pa_array, start=start, stop=stop)
        print(f"  [{start}:{stop}] -> {result.to_pylist()}")
    except Exception as e:
        print(f"  [{start}:{stop}] -> Error: {e}")