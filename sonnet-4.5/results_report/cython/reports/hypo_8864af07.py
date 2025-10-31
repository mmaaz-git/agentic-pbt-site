import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
import tempfile
import os
import string
from Cython.Tempita import Template

@given(st.text(alphabet=string.printable, max_size=200))
@settings(max_examples=100)
def test_from_filename_without_encoding(content):
    # Skip content with template delimiters to avoid template parsing issues
    if '{{' in content or '}}' in content:
        return

    with tempfile.NamedTemporaryFile(mode='w', suffix='.tmpl', delete=False) as f:
        f.write(content)
        filename = f.name

    try:
        template = Template.from_filename(filename)
        result = template.substitute({})
        assert result == content
    finally:
        os.unlink(filename)

if __name__ == "__main__":
    # Run the test
    test_from_filename_without_encoding()