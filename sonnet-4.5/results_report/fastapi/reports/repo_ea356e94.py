from fastapi.security.api_key import APIKeyBase

# Test whitespace-only strings
result = APIKeyBase.check_api_key("   ", auto_error=False)
print(f"Whitespace key result: {repr(result)}")

result2 = APIKeyBase.check_api_key("\t\r\n", auto_error=False)
print(f"Tab/newline key result: {repr(result2)}")

result3 = APIKeyBase.check_api_key("\x85", auto_error=False)
print(f"Unicode NEL result: {repr(result3)}")

result4 = APIKeyBase.check_api_key("", auto_error=False)
print(f"Empty key result: {repr(result4)}")

# Test with auto_error=True to see if exceptions are raised
print("\nWith auto_error=True:")
try:
    result5 = APIKeyBase.check_api_key("   ", auto_error=True)
    print(f"Whitespace with auto_error=True: {repr(result5)} (no exception raised)")
except Exception as e:
    print(f"Whitespace with auto_error=True: Exception raised - {e}")

try:
    result6 = APIKeyBase.check_api_key("", auto_error=True)
    print(f"Empty string with auto_error=True: {repr(result6)} (no exception raised)")
except Exception as e:
    print(f"Empty string with auto_error=True: Exception raised - {e}")