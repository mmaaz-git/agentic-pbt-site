from pandas.io.excel._openpyxl import OpenpyxlWriter

print("=== Testing check_extension method ===")
print("Supported extensions:", OpenpyxlWriter._supported_extensions)
print()

# Test invalid extensions that are substrings
test_cases = ['.l', '.x', '.s', '.m', '.ls', '.xl']
for ext in test_cases:
    try:
        result = OpenpyxlWriter.check_extension(ext)
        print(f"check_extension('{ext}') returned: {result}")
    except ValueError as e:
        print(f"check_extension('{ext}') raised ValueError: {e}")

print()

# Test valid extensions
print("Testing valid extensions:")
valid_cases = ['.xlsx', '.xlsm']
for ext in valid_cases:
    try:
        result = OpenpyxlWriter.check_extension(ext)
        print(f"check_extension('{ext}') returned: {result}")
    except ValueError as e:
        print(f"check_extension('{ext}') raised ValueError: {e}")

print()

# Test completely unrelated extension
print("Testing completely unrelated extension:")
try:
    result = OpenpyxlWriter.check_extension('.pdf')
    print(f"check_extension('.pdf') returned: {result}")
except ValueError as e:
    print(f"check_extension('.pdf') raised ValueError: {e}")