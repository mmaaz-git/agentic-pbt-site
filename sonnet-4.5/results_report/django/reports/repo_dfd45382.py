from django.conf import settings
from django.conf.urls.static import static

# Configure Django settings
settings.configure(DEBUG=True)

# Test with slash-only prefix
prefix = "/"
result = static(prefix)
pattern = result[0].pattern.regex

print(f"Prefix: '{prefix}'")
print(f"Stripped prefix: '{prefix.lstrip('/')}'")
print(f"Regex pattern: {pattern.pattern}")
print()

# Test what URLs this pattern matches
test_paths = ["media/file.jpg", "admin/login", "api/users", ""]
print("Testing URL matches:")
for path in test_paths:
    match = pattern.match(path)
    matches = match is not None
    print(f"  '{path}' matches: {matches}")
    if match:
        print(f"    Captured path: '{match.group('path')}'")