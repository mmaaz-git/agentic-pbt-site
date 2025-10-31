import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

from unittest.mock import Mock

from hypothesis import given, settings, strategies as st

import django
from django.conf import settings as django_settings

if not django_settings.configured:
    django_settings.configure(
        DEBUG=True,
        SECRET_KEY='test-secret-key',
        SECURE_REFERRER_POLICY=None,
        SECURE_HSTS_SECONDS=0,
        SECURE_HSTS_INCLUDE_SUBDOMAINS=False,
        SECURE_HSTS_PRELOAD=False,
        SECURE_CONTENT_TYPE_NOSNIFF=False,
        SECURE_SSL_REDIRECT=False,
        SECURE_SSL_HOST=None,
        SECURE_REDIRECT_EXEMPT=[],
        SECURE_CROSS_ORIGIN_OPENER_POLICY=None,
    )
    django.setup()

from django.middleware.security import SecurityMiddleware
from django.http import HttpResponse, HttpRequest


@given(st.lists(
    st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=50),
    min_size=1,
    max_size=5
))
@settings(max_examples=200)
def test_referrer_policy_whitespace_handling_consistency(policy_values):
    get_response = Mock()

    policy_str = ','.join(policy_values)
    policy_list = list(policy_values)

    django_settings.SECURE_REFERRER_POLICY = policy_str
    middleware_str = SecurityMiddleware(get_response)
    request_str = Mock(spec=HttpRequest)
    request_str.is_secure.return_value = False
    response_str = HttpResponse()
    result_str = middleware_str.process_response(request_str, response_str)
    header_from_str = result_str.get('Referrer-Policy')

    django_settings.SECURE_REFERRER_POLICY = policy_list
    middleware_list = SecurityMiddleware(get_response)
    request_list = Mock(spec=HttpRequest)
    request_list.is_secure.return_value = False
    response_list = HttpResponse()
    result_list = middleware_list.process_response(request_list, response_list)
    header_from_list = result_list.get('Referrer-Policy')

    assert header_from_str == header_from_list

if __name__ == "__main__":
    test_referrer_policy_whitespace_handling_consistency()