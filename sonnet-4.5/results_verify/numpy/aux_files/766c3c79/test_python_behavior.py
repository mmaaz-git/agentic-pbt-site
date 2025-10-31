print("Testing Python's str.replace behavior:")
print("="*50)

# Test with empty string
print("1. Empty string as search pattern:")
s = 'abc'
result = s.replace('', 'X')
print(f"   'abc'.replace('', 'X') = {repr(result)}")

# Test with null character search in string without nulls
print("\n2. Null character search in string without nulls:")
s = 'abc'
result = s.replace('\x00', 'X')
print(f"   'abc'.replace('\\x00', 'X') = {repr(result)}")

# Test with null character in string
print("\n3. Null character search in string with null:")
s = 'a\x00b'
result = s.replace('\x00', 'X')
print(f"   'a\\x00b'.replace('\\x00', 'X') = {repr(result)}")

# Test with empty string
print("\n4. Null character search in empty string:")
s = ''
result = s.replace('\x00', 'X')
print(f"   ''.replace('\\x00', 'X') = {repr(result)}")

# Test null search with count
print("\n5. Null character search with count=1:")
s = 'abc'
result = s.replace('\x00', 'X', 1)
print(f"   'abc'.replace('\\x00', 'X', 1) = {repr(result)}")