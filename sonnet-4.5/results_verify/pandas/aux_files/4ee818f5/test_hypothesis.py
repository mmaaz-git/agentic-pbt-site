from hypothesis import given, strategies as st
from pandas.util._decorators import _format_argument_list


@given(st.lists(st.text()))
def test_format_argument_list_does_not_mutate(args_list):
    original = args_list.copy()
    _format_argument_list(args_list)
    assert args_list == original, f"List was mutated! Original: {original}, After: {args_list}"

# Test with specific failing input
def test_specific_case():
    args_list = ['self', 'arg1', 'arg2']
    original = args_list.copy()
    result = _format_argument_list(args_list)
    print(f"Original: {original}")
    print(f"After: {args_list}")
    print(f"Result: {result}")
    assert args_list == original, f"List was mutated! Original: {original}, After: {args_list}"

if __name__ == "__main__":
    # Test specific case first
    print("Testing specific case ['self', 'arg1', 'arg2']:")
    try:
        test_specific_case()
        print("Specific case passed")
    except AssertionError as e:
        print(f"Specific case failed: {e}")

    print("\n" + "="*50 + "\n")

    # Run hypothesis test
    print("Running hypothesis test...")
    test_format_argument_list_does_not_mutate()