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

from hypothesis import given, strategies as st
from django.templatetags.static import PrefixNode

@given(
    tag_name=st.text(min_size=1, max_size=10),
    varname=st.text(min_size=1, max_size=10)
)
def test_prefix_node_incomplete_as(tag_name, varname):
    class MockToken:
        def __init__(self, contents):
            self.contents = contents

    class MockParser:
        pass

    tokens_2 = [tag_name, 'as']
    token = MockToken(' '.join(tokens_2))
    parser = MockParser()
    PrefixNode.handle_token(parser, token, 'STATIC_URL')

# Run the test
test_prefix_node_incomplete_as()