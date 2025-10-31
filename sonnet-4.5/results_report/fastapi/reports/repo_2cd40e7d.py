from fastapi.openapi.utils import get_openapi

# Test case 1: Empty lists for servers and tags
result = get_openapi(
    title="Test API",
    version="1.0.0",
    servers=[],
    tags=[],
    routes=[]
)

print("Test with empty lists:")
print(f"Has servers: {'servers' in result}")
print(f"Has tags: {'tags' in result}")
print()

# Test case 2: None values for servers and tags
result_none = get_openapi(
    title="Test API",
    version="1.0.0",
    servers=None,
    tags=None,
    routes=[]
)

print("Test with None values:")
print(f"Has servers: {'servers' in result_none}")
print(f"Has tags: {'tags' in result_none}")
print()

# Test case 3: Non-empty lists
result_nonempty = get_openapi(
    title="Test API",
    version="1.0.0",
    servers=[{"url": "http://localhost:8000"}],
    tags=[{"name": "test", "description": "Test tag"}],
    routes=[]
)

print("Test with non-empty lists:")
print(f"Has servers: {'servers' in result_nonempty}")
print(f"Servers value: {result_nonempty.get('servers', 'NOT PRESENT')}")
print(f"Has tags: {'tags' in result_nonempty}")
print(f"Tags value: {result_nonempty.get('tags', 'NOT PRESENT')}")