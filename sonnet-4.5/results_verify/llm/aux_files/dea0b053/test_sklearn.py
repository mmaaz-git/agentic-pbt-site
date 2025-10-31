from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Test with zero vectors
X = [[0, 0, 0]]
Y = [[1, 2, 3]]

print("Testing sklearn's cosine_similarity with zero vector:")
result = cosine_similarity(X, Y)
print(f"Result: {result}")
print(f"Type: {type(result)}")

# Both zero
X = [[0, 0, 0]]
Y = [[0, 0, 0]]
print("\nBoth zero vectors:")
result = cosine_similarity(X, Y)
print(f"Result: {result}")

# Test with numpy.linalg.norm
a = np.array([0, 0, 0])
b = np.array([1, 2, 3])

print("\nNumPy calculation:")
dot = np.dot(a, b)
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)
print(f"Dot product: {dot}")
print(f"Norm a: {norm_a}")
print(f"Norm b: {norm_b}")

if norm_a == 0 or norm_b == 0:
    print("Would cause division by zero")
else:
    print(f"Cosine similarity: {dot / (norm_a * norm_b)}")