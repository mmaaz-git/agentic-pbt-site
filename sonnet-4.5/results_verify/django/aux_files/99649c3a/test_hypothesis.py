import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'django.conf.global_settings'

from hypothesis import given, strategies as st, settings
from django.template.backends.jinja2 import get_exception_info


@settings(max_examples=500)
@given(
    st.text(min_size=0, max_size=100),
    st.integers(min_value=1, max_value=200),
    st.text(max_size=50)
)
def test_get_exception_info_no_crash(source, lineno, message):
    class MockException:
        def __init__(self):
            self.lineno = lineno
            self.source = source
            self.filename = 'test.html'
            self.message = message

    exc = MockException()
    info = get_exception_info(exc)
    assert info['line'] == lineno

if __name__ == "__main__":
    test_get_exception_info_no_crash()