import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume
import pytest
from Cython.Tempita import Template
from Cython.Tempita._tempita import TemplateError

@given(st.integers(min_value=1, max_value=5))
def test_template_rejects_duplicate_else(num_else_clauses):
    assume(num_else_clauses >= 2)

    else_blocks = ''.join([f"{{{{else}}}}{i}\n" for i in range(num_else_clauses)])
    content = f"{{{{if False}}}}\nA\n{else_blocks}{{{{endif}}}}"

    print(f"Testing with {num_else_clauses} else clauses")
    print(f"Template content: {content}")

    with pytest.raises(TemplateError):
        Template(content)

# Run the test without using hypothesis
def test_duplicate_else_manually(num_else_clauses):
    else_blocks = ''.join([f"{{{{else}}}}{i}\n" for i in range(num_else_clauses)])
    content = f"{{{{if False}}}}\nA\n{else_blocks}{{{{endif}}}}"

    print(f"Testing with {num_else_clauses} else clauses")
    print(f"Template content: {content}")

    try:
        template = Template(content)
        result = template.substitute({})
        print(f"Template created successfully. Result: {result}")
        print("ERROR: Template should have raised TemplateError but didn't!")
        return False
    except TemplateError as e:
        print(f"Template correctly raised TemplateError: {e}")
        return True

if __name__ == "__main__":
    # Try with different numbers of else clauses
    for num in [2, 3, 4, 5]:
        print(f"\n--- Testing with {num} else clauses ---")
        test_duplicate_else_manually(num)