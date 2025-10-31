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

    assert header_from_str == header_from_list, f"String: {header_from_str!r}, List: {header_from_list!r}, Values: {policy_values!r}"


def manual_test(policy_values):
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

    return header_from_str, header_from_list


if __name__ == "__main__":
    # Test with the specific failing input
    print("Testing with failing input: [' ']")
    str_result, list_result = manual_test([' '])
    print(f"String input ' ': {str_result!r}")
    print(f"List input [' ']: {list_result!r}")
    if str_result == list_result:
        print("✓ Test passed - results match")
    else:
        print("✗ Test failed - results don't match")

    # Test with more examples
    print("\nTesting with [' no-referrer ', '  strict-origin']")
    str_result, list_result = manual_test([' no-referrer ', '  strict-origin'])
    print(f"String input ' no-referrer ,  strict-origin': {str_result!r}")
    print(f"List input [' no-referrer ', '  strict-origin']: {list_result!r}")
    if str_result == list_result:
        print("✓ Test passed - results match")
    else:
        print("✗ Test failed - results don't match")