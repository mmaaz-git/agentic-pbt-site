from llm.default_plugins.openai_models import not_nulls

# Test case from the bug report
data = {'': None}
try:
    result = not_nulls(data)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Additional test with non-empty keys
data2 = {'key1': 'value1', 'key2': None, 'key3': 42}
try:
    result2 = not_nulls(data2)
    print(f"Result2: {result2}")
except Exception as e:
    print(f"Error2: {type(e).__name__}: {e}")