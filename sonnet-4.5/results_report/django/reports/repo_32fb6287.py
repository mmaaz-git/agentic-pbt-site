import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

import re
from unittest.mock import patch
import django
from django.conf import settings

# Configure Django settings
settings.configure(DEBUG=True)

from django.conf.urls.static import static

# Test the bug with "/" prefix
result = static("/")

pattern_obj = result[0]
regex = pattern_obj.pattern._regex

print(f"Pattern regex: {regex}")

test_urls = ["foo", "bar/baz", "media/image.png", "admin/login"]
for url in test_urls:
    match = re.match(regex, url)
    if match:
        print(f"'{url}' matches (captured: {match.group('path')})")