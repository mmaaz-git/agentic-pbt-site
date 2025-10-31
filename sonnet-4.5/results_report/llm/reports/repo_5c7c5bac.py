import llm

# Test case that crashes with zero vector
result = llm.cosine_similarity([0.0, 0.0], [1.0, 2.0])
print(f"Result: {result}")