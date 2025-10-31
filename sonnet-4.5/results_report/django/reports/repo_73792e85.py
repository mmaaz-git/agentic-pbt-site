import os
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

from django.conf import settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        STATIC_URL='/static/',
        TEMPLATES=[{
            'BACKEND': 'django.template.backends.django.DjangoTemplates',
        }],
    )

import django
django.setup()

from django.templatetags.static import PrefixNode

class MockToken:
    def __init__(self, contents):
        self.contents = contents

class MockParser:
    pass

# Test the bug case: incomplete 'as' syntax without variable name
token = MockToken('get_static_prefix as')
parser = MockParser()

try:
    result = PrefixNode.handle_token(parser, token, 'STATIC_URL')
    print(f"Result: {result}")
except Exception as e:
    print(f"Exception: {type(e).__name__}: {e}")