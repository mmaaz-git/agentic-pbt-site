from django.conf import settings
from django.conf.urls.static import static
from django.core.exceptions import ImproperlyConfigured

# Configure Django settings
settings.configure(DEBUG=True, SECRET_KEY='test')

print("=== Testing static('/') behavior ===")
result = static('/')
if result:
    print(f"Pattern: {result[0].pattern.regex.pattern}")
    print(f"Number of patterns returned: {len(result)}")
else:
    print("No patterns returned")

print("\n=== Testing static('') behavior ===")
try:
    static('')
    print("No error raised for empty string")
except ImproperlyConfigured as e:
    print(f"Empty string raises: {e}")

print("\n=== Testing static('//') behavior ===")
result2 = static('//')
if result2:
    print(f"Pattern: {result2[0].pattern.regex.pattern}")
else:
    print("No patterns returned")

print("\n=== Testing static('///') behavior ===")
result3 = static('///')
if result3:
    print(f"Pattern: {result3[0].pattern.regex.pattern}")
else:
    print("No patterns returned")

print("\n=== Testing normal prefix '/media/' ===")
result4 = static('/media/')
if result4:
    print(f"Pattern: {result4[0].pattern.regex.pattern}")
else:
    print("No patterns returned")

print("\n=== Testing prefix.lstrip('/') results ===")
test_cases = ['/', '//', '///', '/media/', 'media/', '']
for prefix in test_cases:
    stripped = prefix.lstrip('/')
    print(f"'{prefix}'.lstrip('/') = '{stripped}'")