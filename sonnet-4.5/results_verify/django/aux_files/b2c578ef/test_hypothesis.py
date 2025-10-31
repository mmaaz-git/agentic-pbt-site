import sys
from unittest.mock import Mock, patch

sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st, settings
from django.core.checks.security.base import check_cross_origin_opener_policy


@given(st.lists(st.text(), min_size=0, max_size=3))
@settings(max_examples=100)
def test_cross_origin_opener_policy_with_list(value):
    mock_settings = Mock()
    mock_settings.SECURE_CROSS_ORIGIN_OPENER_POLICY = value

    with patch('django.core.checks.security.base.settings', mock_settings):
        with patch('django.core.checks.security.base._security_middleware', return_value=True):
            result = check_cross_origin_opener_policy(None)

if __name__ == "__main__":
    # Run the test
    test_cross_origin_opener_policy_with_list()