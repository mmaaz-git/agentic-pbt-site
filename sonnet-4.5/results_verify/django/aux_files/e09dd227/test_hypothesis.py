import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')
import django
from django.conf import settings
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY='test-key-for-testing-minimum-length-of-fifty-chars!!',
        MIDDLEWARE=['django.middleware.security.SecurityMiddleware'],
    )
    django.setup()

from hypothesis import given, strategies as st
from django.test import override_settings
from django.core.checks.security.base import check_referrer_policy


@given(st.one_of(st.none(), st.lists(st.sampled_from([
    "no-referrer", "no-referrer-when-downgrade", "origin",
    "origin-when-cross-origin", "same-origin", "strict-origin",
    "strict-origin-when-cross-origin", "unsafe-url"
]), min_size=0, max_size=5)))
def test_referrer_policy_none_vs_empty_consistency(policy_value):
    with override_settings(SECURE_REFERRER_POLICY=policy_value):
        result = check_referrer_policy(None)

    if policy_value is None or policy_value == []:
        assert len(result) > 0, \
            f"Empty policy should trigger warning. Policy: {policy_value}, Result: {result}"

if __name__ == "__main__":
    test_referrer_policy_none_vs_empty_consistency()
    print("Test completed")