import llm

# Test case that should cause ZeroDivisionError
result = llm.cosine_similarity([0.0], [0.0])
print(f"Result: {result}")