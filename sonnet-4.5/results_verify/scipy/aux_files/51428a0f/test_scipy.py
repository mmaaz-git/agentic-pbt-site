from scipy.datasets import face

# Test with gray=1 (truthy non-boolean value)
result_1 = face(gray=1)
print(f"face(gray=1) shape: {result_1.shape}")
print(f"Expected: (768, 1024) for grayscale")
print(f"Actual: {result_1.shape}")

# Test with gray=True
result_true = face(gray=True)
print(f"\nface(gray=True) shape: {result_true.shape}")

# Test with other truthy values
result_str = face(gray="yes")
print(f"\nface(gray='yes') shape: {result_str.shape}")

result_list = face(gray=[1,2,3])
print(f"face(gray=[1,2,3]) shape: {result_list.shape}")

# Test with falsy non-boolean values
result_0 = face(gray=0)
print(f"\nface(gray=0) shape: {result_0.shape}")

result_false = face(gray=False)
print(f"face(gray=False) shape: {result_false.shape}")