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

from hypothesis import given, strategies as st
from django.core.checks.registry import CheckRegistry
from django.core.checks import Error


@given(
    st.text(min_size=1, max_size=10),
    st.text(min_size=1, max_size=10)
)
def test_registry_multiple_registration_different_tags(tag1, tag2):
    """Registering the same check with different tags should preserve all tags"""
    registry = CheckRegistry()

    def my_check(app_configs, **kwargs):
        return [Error("Test error")]

    registry.register(my_check, tag1)
    registry.register(my_check, tag2)

    all_errors = registry.run_checks()
    tag1_errors = registry.run_checks(tags=[tag1])
    tag2_errors = registry.run_checks(tags=[tag2])

    assert len(all_errors) >= 1
    if tag1 != tag2:
        assert len(tag1_errors) >= 1, f"Check registered with tag1={tag1} should be callable with that tag"
        assert len(tag2_errors) >= 1, f"Check registered with tag2={tag2} should be callable with that tag"

if __name__ == "__main__":
    test_registry_multiple_registration_different_tags()