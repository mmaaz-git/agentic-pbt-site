import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')
import django
from django.conf import settings
settings.configure(
    DEBUG=True,
    SECRET_KEY='test' * 20,
    MIDDLEWARE=['django.middleware.security.SecurityMiddleware'],
)
django.setup()

from django.test import override_settings
from django.core.checks.security.base import check_referrer_policy

with override_settings(SECURE_REFERRER_POLICY=None):
    result_none = check_referrer_policy(None)
    print(f"None: {len(result_none)} warnings")
    if result_none:
        print(f"  Warning: {result_none[0]}")

with override_settings(SECURE_REFERRER_POLICY=[]):
    result_empty = check_referrer_policy(None)
    print(f"[]:   {len(result_empty)} warnings")
    if result_empty:
        print(f"  Warning: {result_empty[0]}")