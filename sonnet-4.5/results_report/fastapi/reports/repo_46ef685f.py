import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')

from fastapi.exceptions import ResponseValidationError

# Test with single error
exc = ResponseValidationError(["field validation failed"])
print("Single error case:")
print(str(exc))
print()

# Test with multiple errors
exc_multiple = ResponseValidationError(["field1 validation failed", "field2 validation failed"])
print("Multiple errors case:")
print(str(exc_multiple))
print()

# Test with zero errors (edge case)
exc_zero = ResponseValidationError([])
print("Zero errors case:")
print(str(exc_zero))