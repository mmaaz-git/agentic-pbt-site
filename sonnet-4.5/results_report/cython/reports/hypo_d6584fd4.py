import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import pytest
from hypothesis import given, assume, strategies as st
from Cython.Tempita import Template, TemplateError

@given(st.integers(min_value=1, max_value=5))
def test_template_rejects_duplicate_else(num_else_clauses):
    """Test that templates with duplicate else clauses are rejected."""
    assume(num_else_clauses >= 2)

    else_blocks = ''.join([f"{{{{else}}}}{i}\n" for i in range(num_else_clauses)])
    content = f"{{{{if False}}}}\nA\n{else_blocks}{{{{endif}}}}"

    print(f"\nTesting with {num_else_clauses} else clauses:")
    print(f"Template content: {repr(content)}")

    with pytest.raises(TemplateError):
        Template(content)

if __name__ == "__main__":
    # Run the test with a specific failing case
    print("Running property-based test for duplicate else clauses...")
    print("=" * 60)

    def run_test():
        for num_else in [2, 3, 4]:
            print(f"\nTesting with {num_else} else clauses:")
            else_blocks = ''.join([f"{{{{else}}}}{i}\n" for i in range(num_else)])
            content = f"{{{{if False}}}}\nA\n{else_blocks}{{{{endif}}}}"
            print(f"Template content: {repr(content)}")

            try:
                template = Template(content)
                print(f"ERROR: Template was accepted (should have raised TemplateError)")
                result = template.substitute({})
                print(f"Result when executed: {repr(result)}")
            except TemplateError as e:
                print(f"SUCCESS: Template was rejected with error: {e}")

    try:
        run_test()
        print("\n" + "=" * 60)
        print("Test FAILED: Templates with duplicate else were accepted")
    except AssertionError as e:
        print(f"Test FAILED: Template with duplicate else was accepted when it should have been rejected")
        # Show what actually happens
        content = "{{if False}}\nA\n{{else}}0\n{{else}}1\n{{endif}}"
        print(f"\nActual behavior with template: {repr(content)}")
        template = Template(content)
        result = template.substitute({})
        print(f"Result: {repr(result)}")
        print("\nThis demonstrates that invalid syntax is silently accepted.")
    except Exception as e:
        print(f"Unexpected error: {e}")