from django.conf import settings
from django.conf.urls.static import static

settings.configure(DEBUG=True)

prefix = "/"
result = static(prefix)
pattern = result[0].pattern.regex

print(f"Prefix: '{prefix}'")
print(f"Regex pattern: {pattern.pattern}")

test_paths = ["media/file.jpg", "admin/login", "api/users", ""]
for path in test_paths:
    matches = pattern.match(path) is not None
    print(f"  '{path}' matches: {matches}")

# Let's also test with multiple slashes
print("\nTesting with multiple slashes:")
for num_slashes in [1, 2, 3, 10]:
    prefix = '/' * num_slashes
    result = static(prefix)
    if result:
        pattern = result[0].pattern.regex.pattern
        print(f"  Prefix: '{prefix}' -> Pattern: {pattern}")
        print(f"    After lstrip('/'): '{prefix.lstrip('/')}'")

# Test with valid prefixes for comparison
print("\nTesting with valid prefixes:")
for prefix in ["/media/", "static/", "/assets/"]:
    result = static(prefix)
    if result:
        pattern = result[0].pattern.regex.pattern
        print(f"  Prefix: '{prefix}' -> Pattern: {pattern}")
        print(f"    After lstrip('/'): '{prefix.lstrip('/')}'")