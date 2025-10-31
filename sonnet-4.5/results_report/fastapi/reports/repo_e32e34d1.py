from fastapi.openapi.utils import get_openapi

# Test case showing the bug with empty string
result = get_openapi(
    title="Test API",
    version="1.0.0",
    terms_of_service="",
    routes=[]
)

print(f"Has termsOfService: {'termsOfService' in result['info']}")
print(f"Info object: {result['info']}")

# Test case with None (should not include termsOfService)
result_none = get_openapi(
    title="Test API",
    version="1.0.0",
    terms_of_service=None,
    routes=[]
)

print(f"\nWith None - Has termsOfService: {'termsOfService' in result_none['info']}")
print(f"Info object: {result_none['info']}")

# Test case with non-empty string (should include termsOfService)
result_value = get_openapi(
    title="Test API",
    version="1.0.0",
    terms_of_service="https://example.com/terms",
    routes=[]
)

print(f"\nWith value - Has termsOfService: {'termsOfService' in result_value['info']}")
print(f"Info object: {result_value['info']}")