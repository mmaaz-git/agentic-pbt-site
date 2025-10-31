import sys
from unittest.mock import Mock, patch

sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from django.core.checks.security.base import check_cross_origin_opener_policy

print("Testing with empty list...")
mock_settings = Mock()
mock_settings.SECURE_CROSS_ORIGIN_OPENER_POLICY = []

try:
    with patch('django.core.checks.security.base.settings', mock_settings):
        with patch('django.core.checks.security.base._security_middleware', return_value=True):
            result = check_cross_origin_opener_policy(None)
            print(f"Result: {result}")
except TypeError as e:
    print(f"ERROR: {e}")

print("\nTesting with non-empty list...")
mock_settings.SECURE_CROSS_ORIGIN_OPENER_POLICY = ['same-origin']

try:
    with patch('django.core.checks.security.base.settings', mock_settings):
        with patch('django.core.checks.security.base._security_middleware', return_value=True):
            result = check_cross_origin_opener_policy(None)
            print(f"Result: {result}")
except TypeError as e:
    print(f"ERROR: {e}")

print("\nTesting with dict (another unhashable type)...")
mock_settings.SECURE_CROSS_ORIGIN_OPENER_POLICY = {'policy': 'same-origin'}

try:
    with patch('django.core.checks.security.base.settings', mock_settings):
        with patch('django.core.checks.security.base._security_middleware', return_value=True):
            result = check_cross_origin_opener_policy(None)
            print(f"Result: {result}")
except TypeError as e:
    print(f"ERROR: {e}")

print("\nTesting with valid string value...")
mock_settings.SECURE_CROSS_ORIGIN_OPENER_POLICY = 'same-origin'

try:
    with patch('django.core.checks.security.base.settings', mock_settings):
        with patch('django.core.checks.security.base._security_middleware', return_value=True):
            result = check_cross_origin_opener_policy(None)
            print(f"Result: {result}")
except TypeError as e:
    print(f"ERROR: {e}")

print("\nTesting with invalid string value...")
mock_settings.SECURE_CROSS_ORIGIN_OPENER_POLICY = 'invalid-value'

try:
    with patch('django.core.checks.security.base.settings', mock_settings):
        with patch('django.core.checks.security.base._security_middleware', return_value=True):
            result = check_cross_origin_opener_policy(None)
            print(f"Result: {result}")
except TypeError as e:
    print(f"ERROR: {e}")