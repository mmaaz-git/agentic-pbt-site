import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'test_settings')

from hypothesis import given, strategies as st, settings
from django.template.backends.jinja2 import get_exception_info


@given(
    lineno=st.integers(min_value=1),
    source=st.text(),
    filename=st.text(min_size=1),
    message=st.text(),
)
@settings(max_examples=500)
def test_get_exception_info_doesnt_crash(lineno, source, filename, message):
    class MockException:
        pass

    exc = MockException()
    exc.lineno = lineno
    exc.filename = filename
    exc.message = message
    exc.source = source

    result = get_exception_info(exc)

    assert isinstance(result, dict)

# Run the test
test_get_exception_info_doesnt_crash()
print("Test completed successfully")