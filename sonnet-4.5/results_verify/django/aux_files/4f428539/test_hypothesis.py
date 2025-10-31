import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')
from hypothesis import given, strategies as st, example
from django.core.management.utils import handle_extensions

@given(st.lists(st.text(alphabet=' ,', min_size=1, max_size=20), min_size=1, max_size=5))
@example([','])
@example(['html,,css'])
@example([''])
def test_handle_extensions_no_single_dot(separator_strings):
    print(f"Testing input: {separator_strings}")
    result = handle_extensions(separator_strings)
    print(f"Result: {result}")
    assert '.' not in result, f"Invalid extension '.' should not be in result for input {separator_strings}, got {result}"

if __name__ == "__main__":
    test_handle_extensions_no_single_dot()