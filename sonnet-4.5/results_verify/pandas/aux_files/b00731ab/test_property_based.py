import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given
import hypothesis.strategies as st
import pytest
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

if __name__ == "__main__":
    # Run the test directly without hypothesis
    val1, val2, val3 = 1, 2, 3
    content = f"""
{{{{if False}}}}
  {val1}
{{{{else}}}}
  {val2}
{{{{else}}}}
  {val3}
{{{{endif}}}}
"""
    try:
        template = Template(content)
        print(f"ERROR: Template was created without raising an error!")
        print(f"Result: {template.substitute({})}")
    except TemplateError as e:
        if "duplicate" in str(e).lower() or "multiple" in str(e).lower():
            print("Test passed! TemplateError raised as expected")
        else:
            print(f"TemplateError raised but with unexpected message: {e}")