import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume
from Cython.Tempita import Template

@given(st.text(min_size=1, max_size=100))
def test_template_bytes_content_handling(value):
    assume('\x00' not in value)

    content_bytes = value.encode('utf-8')
    template = Template(content_bytes)

    result = template.substitute({})
    assert isinstance(result, bytes) or isinstance(result, str)

# Run the test
test_template_bytes_content_handling()