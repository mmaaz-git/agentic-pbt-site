from hypothesis import given, strategies as st, assume, settings
from Cython.Build.Dependencies import parse_list

@given(st.lists(st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=1)))
@settings(max_examples=500)
def test_parse_list_bracket_format_roundtrip(items):
    assume(all(item for item in items))
    assume(all(',' not in item and '"' not in item and "'" not in item for item in items))

    input_str = '[' + ', '.join(items) + ']'
    result = parse_list(input_str)
    assert result == items, f"Expected {items}, got {result} for input {repr(input_str)}"

if __name__ == "__main__":
    # Test the specific failing case mentioned
    print("Testing specific case: [' ']")
    result = parse_list("[' ']")
    print(f"Result: {repr(result)}")
    print(f"Expected: [' ']")
    print(f"Match: {result == [' ']}")

    # Run hypothesis tests
    print("\nRunning property-based tests...")
    test_parse_list_bracket_format_roundtrip()
    print("All tests passed!")