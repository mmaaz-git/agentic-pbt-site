import numpy as np

print("Testing numpy.linspace behavior with num=1...")
print()

# Test with endpoint=True
print("numpy.linspace(0, 1, 1, endpoint=True):")
result = np.linspace(0, 1, 1, endpoint=True)
print(f"  Result: {result}")
print(f"  Shape: {result.shape}")
print(f"  Value: {result[0]}")
print()

# Test with endpoint=False
print("numpy.linspace(0, 1, 1, endpoint=False):")
result = np.linspace(0, 1, 1, endpoint=False)
print(f"  Result: {result}")
print(f"  Shape: {result.shape}")
print(f"  Value: {result[0]}")
print()

# Test with different start/stop values
print("numpy.linspace(5, 10, 1, endpoint=True):")
result = np.linspace(5, 10, 1, endpoint=True)
print(f"  Result: {result}")
print(f"  Value: {result[0]}")
print()

print("numpy.linspace(5, 10, 1, endpoint=False):")
result = np.linspace(5, 10, 1, endpoint=False)
print(f"  Result: {result}")
print(f"  Value: {result[0]}")