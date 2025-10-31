import os
import sys

# Add Django to path
django_path = '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages'
sys.path.insert(0, django_path)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

import django
from django.conf import settings

if not settings.configured:
    settings.configure(DEBUG=True, SECRET_KEY='test')
    django.setup()

from django.conf.urls.static import static

print("Testing static() with different prefixes:")
print("-" * 50)

# Test with "/" prefix
print("\nTest 1: prefix='/'")
result = static("/")
if result:
    pattern = result[0].pattern.regex
    print(f"Pattern: {pattern.pattern}")
    print(f"Pattern matches empty string '': {bool(pattern.match(''))}")
    print(f"Pattern matches 'admin/': {bool(pattern.match('admin/'))}")
    print(f"Pattern matches 'api/users/123': {bool(pattern.match('api/users/123'))}")
    print(f"Pattern matches 'any/arbitrary/url': {bool(pattern.match('any/arbitrary/url'))}")
else:
    print("Result is empty list")

# Test with normal prefix
print("\n" + "-" * 50)
print("\nTest 2: prefix='/media/'")
result2 = static("/media/")
if result2:
    pattern2 = result2[0].pattern.regex
    print(f"Pattern: {pattern2.pattern}")
    print(f"Pattern matches empty string '': {bool(pattern2.match(''))}")
    print(f"Pattern matches 'admin/': {bool(pattern2.match('admin/'))}")
    print(f"Pattern matches 'media/file.jpg': {bool(pattern2.match('media/file.jpg'))}")
    print(f"Pattern matches 'api/users/123': {bool(pattern2.match('api/users/123'))}")
else:
    print("Result is empty list")

# Test with empty prefix (should raise error)
print("\n" + "-" * 50)
print("\nTest 3: prefix='' (should raise error)")
try:
    result3 = static("")
    print("No error raised - this is unexpected!")
except Exception as e:
    print(f"Error raised as expected: {e}")

# Test with multiple slashes
print("\n" + "-" * 50)
print("\nTest 4: prefix='//'")
result4 = static("//")
if result4:
    pattern4 = result4[0].pattern.regex
    print(f"Pattern: {pattern4.pattern}")
    print(f"Pattern matches 'any/url': {bool(pattern4.match('any/url'))}")
else:
    print("Result is empty list")