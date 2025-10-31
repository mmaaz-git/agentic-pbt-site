import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, assume, strategies as st
from Cython.Tempita import Template

# First, let's test with the property-based test
@given(st.text(alphabet='0123456789+-*/', min_size=1, max_size=10))
def test_syntaxerror_includes_position(expr):
    assume('{{' not in expr and '}}' not in expr)

    content = f"Line 1\nLine 2\n{{{{{expr}}}}}"
    template = Template(content)

    try:
        template.substitute({})
    except SyntaxError as e:
        error_msg = str(e)
        print(f"Expression: {expr}")
        print(f"SyntaxError: {error_msg}")
        assert 'line' in error_msg.lower() and 'column' in error_msg.lower(), f"Position info missing for expression: {expr}"

# Test directly without hypothesis wrapper for specific cases
print("Testing with invalid expressions:")
test_cases = ["/", "**", "+++", "+*", "*/", "//", "--", "^^"]
for expr in test_cases:
    if '{{' not in expr and '}}' not in expr:
        content = f"Line 1\nLine 2\n{{{{{expr}}}}}"
        template = Template(content)

        try:
            template.substitute({})
            print(f"✓ Expression {expr} was valid (no error)")
        except SyntaxError as e:
            error_msg = str(e)
            print(f"Expression: {expr}")
            print(f"  SyntaxError: {error_msg}")
            if 'line' in error_msg.lower() and 'column' in error_msg.lower():
                print(f"  ✓ Has position info")
            else:
                print(f"  ✗ Missing position info")
        except Exception as e:
            print(f"  Other error for {expr}: {e}")