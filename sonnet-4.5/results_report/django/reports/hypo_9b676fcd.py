import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from django.template.backends.jinja2 import get_exception_info

class MockJinja2Exception:
    def __init__(self, lineno, source, message, filename):
        self.lineno = lineno
        self.source = source
        self.message = message
        self.filename = filename

@given(
    st.integers(min_value=1, max_value=10000),
    st.text(min_size=0, max_size=1000),
    st.text(max_size=100)
)
def test_get_exception_info_total_lines(lineno, source, message):
    exc = MockJinja2Exception(
        lineno=lineno,
        source=source,
        message=message,
        filename="test.html"
    )

    info = get_exception_info(exc)

    if source.strip():
        expected_total = len(source.strip().split('\n'))
    else:
        expected_total = 0

    assert info['total'] == expected_total, f"Expected {expected_total} lines but got {info['total']} for source: {repr(source)}"

# Run the test
if __name__ == "__main__":
    test_get_exception_info_total_lines()