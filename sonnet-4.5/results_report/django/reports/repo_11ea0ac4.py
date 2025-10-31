import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

# Create minimal Django settings
with open('test_settings.py', 'w') as f:
    f.write('''
SECRET_KEY = 'test-secret-key'
DEBUG = True
INSTALLED_APPS = []
''')

from django.core.checks.registry import CheckRegistry
from django.core.checks import Error

registry = CheckRegistry()

def my_check(app_configs, **kwargs):
    return [Error("Test error")]

# Register the same check function with 'database' tag
registry.register(my_check, 'database')
print(f"After first registration with 'database' tag:")
print(f"  check.tags = {my_check.tags}")
print(f"  Tags available in registry: {registry.tags_available()}")

# Register the same check function again with 'security' tag
registry.register(my_check, 'security')
print(f"\nAfter second registration with 'security' tag:")
print(f"  check.tags = {my_check.tags}")
print(f"  Tags available in registry: {registry.tags_available()}")

# Try to run checks with the 'database' tag
database_errors = registry.run_checks(tags=['database'])
print(f"\nRunning checks with 'database' tag:")
print(f"  Number of errors returned: {len(database_errors)}")

# Try to run checks with the 'security' tag
security_errors = registry.run_checks(tags=['security'])
print(f"\nRunning checks with 'security' tag:")
print(f"  Number of errors returned: {len(security_errors)}")

# Run all checks without filtering by tags
all_errors = registry.run_checks()
print(f"\nRunning all checks (no tag filter):")
print(f"  Number of errors returned: {len(all_errors)}")

print("\n" + "="*50)
print("EXPECTED BEHAVIOR:")
print("  - Both 'database' and 'security' tags should work")
print("  - Each tag should return 1 error when used")
print("\nACTUAL BEHAVIOR:")
print(f"  - 'database' tag returns {len(database_errors)} errors (expected: 1)")
print(f"  - 'security' tag returns {len(security_errors)} errors (expected: 1)")
print(f"  - The 'database' tag was silently overwritten!")