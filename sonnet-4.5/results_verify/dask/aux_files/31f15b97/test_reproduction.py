import dask.utils

print("Testing parse_bytes with empty string:")
result = dask.utils.parse_bytes('')
print(f"parse_bytes('') = {result}")

print("\nTesting parse_bytes with carriage return:")
result = dask.utils.parse_bytes('\r')
print(f"parse_bytes('\\r') = {result}")

print("\nTesting parse_bytes with newline:")
result = dask.utils.parse_bytes('\n')
print(f"parse_bytes('\\n') = {result}")

print("\nTesting parse_bytes with tab:")
result = dask.utils.parse_bytes('\t')
print(f"parse_bytes('\\t') = {result}")

print("\nTesting parse_bytes with space:")
result = dask.utils.parse_bytes(' ')
print(f"parse_bytes(' ') = {result}")

print("\nTesting parse_bytes with multiple spaces:")
result = dask.utils.parse_bytes('   ')
print(f"parse_bytes('   ') = {result}")

print("\nTesting parse_bytes with valid input for comparison:")
result = dask.utils.parse_bytes('5kb')
print(f"parse_bytes('5kb') = {result}")

print("\nTesting parse_bytes with invalid input (should raise ValueError):")
try:
    result = dask.utils.parse_bytes('5 foos')
    print(f"parse_bytes('5 foos') = {result}")
except ValueError as e:
    print(f"parse_bytes('5 foos') raised ValueError: {e}")