import llm

a = [0.0, 0.0, 0.0]
b = [1.0, 2.0, 3.0]

try:
    result = llm.cosine_similarity(a, b)
    print(f"Result: {result}")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError: {e}")