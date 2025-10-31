import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import pytest
from hypothesis import given
import hypothesis.strategies as st
from Cython.Tempita import Template, TemplateError

@given(st.integers(), st.integers(), st.integers())
def test_template_rejects_duplicate_else(val1, val2, val3):
    content = f"""
{{{{if False}}}}
  {val1}
{{{{else}}}}
  {val2}
{{{{else}}}}
  {val3}
{{{{endif}}}}
"""
    with pytest.raises(TemplateError, match=r"duplicate.*else|multiple.*else"):
        template = Template(content)

# Run the test
test_template_rejects_duplicate_else()