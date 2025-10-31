import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

from django.conf import settings as django_settings

if not django_settings.configured:
    django_settings.configure(
        DEBUG=False,
        SECRET_KEY='test-key-' + 'x' * 50,
        MIDDLEWARE=['django.middleware.security.SecurityMiddleware'],
        INSTALLED_APPS=[],
        ALLOWED_HOSTS=['*'],
    )
    django.setup()

from hypothesis import given, strategies as st, settings, assume
from django.core.checks.security import base

@given(
    st.sampled_from(list(base.CROSS_ORIGIN_OPENER_POLICY_VALUES)),
    st.text(alphabet=' \t\n', min_size=0, max_size=5)
)
@settings(max_examples=50)
def test_cross_origin_opener_policy_no_whitespace_handling(valid_value, whitespace):
    assume(len(whitespace) > 0)

    django_settings.MIDDLEWARE = ['django.middleware.security.SecurityMiddleware']

    django_settings.SECURE_CROSS_ORIGIN_OPENER_POLICY = valid_value
    result_clean = base.check_cross_origin_opener_policy(None)

    django_settings.SECURE_CROSS_ORIGIN_OPENER_POLICY = valid_value + whitespace
    result_with_whitespace = base.check_cross_origin_opener_policy(None)

    assert result_clean == result_with_whitespace, f"Results differ for '{valid_value}' with whitespace '{repr(whitespace)}'"

if __name__ == "__main__":
    test_cross_origin_opener_policy_no_whitespace_handling()