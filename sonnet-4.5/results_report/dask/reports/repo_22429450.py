from dask.utils import parse_bytes

# Test negative values with various units
print("Testing parse_bytes with negative values:")
print(f"parse_bytes('-128MiB') = {parse_bytes('-128MiB')}")
print(f"parse_bytes(-100) = {parse_bytes(-100)}")
print(f"parse_bytes('-5kB') = {parse_bytes('-5kB')}")
print(f"parse_bytes('-1B') = {parse_bytes('-1B')}")
print(f"parse_bytes('-1GB') = {parse_bytes('-1GB')}")
print(f"parse_bytes(-1024) = {parse_bytes(-1024)}")

# Show that these negative values are semantically incorrect
print("\nByte sizes should represent amounts of data, which cannot be negative.")
print("Negative byte sizes make no semantic sense - you cannot have -5MB of RAM or -100KB file size.")