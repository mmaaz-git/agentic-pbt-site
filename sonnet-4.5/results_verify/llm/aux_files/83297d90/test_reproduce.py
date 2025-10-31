import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

from django.core.checks.registry import CheckRegistry
from django.core.checks import Error

registry = CheckRegistry()

def my_check(app_configs, **kwargs):
    return [Error("Test error")]

registry.register(my_check, 'database')
print(f"After first registration, tags: {my_check.tags}")

registry.register(my_check, 'security')
print(f"After second registration, tags: {my_check.tags}")

database_errors = registry.run_checks(tags=['database'])
security_errors = registry.run_checks(tags=['security'])

print(f"Database tag errors: {len(database_errors)}")
print(f"Security tag errors: {len(security_errors)}")

assert len(database_errors) == 0
assert len(security_errors) == 1

print("Assertions passed - bug confirmed!")