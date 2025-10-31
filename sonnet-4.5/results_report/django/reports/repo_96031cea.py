from django.http.cookie import parse_cookie

# Test case 1: Multiple cookies with whitespace-only names
cookie_string = " =first; \t=second; \n=third"
result = parse_cookie(cookie_string)

print("Test case 1: Multiple cookies with whitespace-only names")
print(f"Input:  {cookie_string!r}")
print(f"Output: {result}")
print(f"Expected: 3 separate cookies")
print(f"Actual:   {len(result)} cookie(s)")
print()

# Test case 2: Mix of whitespace-only and normal cookies
cookie_string2 = "normal=value1; \t=whitespace_tab; valid=value2; \n=whitespace_newline"
result2 = parse_cookie(cookie_string2)

print("Test case 2: Mix of whitespace-only and normal cookies")
print(f"Input:  {cookie_string2!r}")
print(f"Output: {result2}")
print(f"Expected: 4 cookies (2 normal, 2 whitespace)")
print(f"Actual:   {len(result2)} cookie(s)")
print()

# Test case 3: Different whitespace characters colliding
cookie_string3 = "\r=carriage_return; \n=newline; \t=tab;  =space"
result3 = parse_cookie(cookie_string3)

print("Test case 3: Different whitespace characters all colliding")
print(f"Input:  {cookie_string3!r}")
print(f"Output: {result3}")
print(f"Expected: 4 different cookies")
print(f"Actual:   {len(result3)} cookie(s)")
print()

# Demonstrate the data loss explicitly
print("Data Loss Summary:")
print("- All whitespace-only cookie names get stripped to empty string ''")
print("- Dictionary can only hold one value per key")
print("- Only the last cookie with whitespace-only name survives")