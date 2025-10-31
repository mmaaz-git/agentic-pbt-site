from hypothesis import given, strategies as st, settings, example
from pydantic.alias_generators import to_pascal

# Run the specific failing case from the bug report
field_name = 'A_A'
once = to_pascal(field_name)
twice = to_pascal(once)
print(f"Testing 'A_A': once='{once}', twice='{twice}', idempotent={once == twice}")

# Now run hypothesis with more specific strategies
@given(st.text(alphabet=st.characters(whitelist_categories=['Lu', 'Ll'], min_codepoint=65, max_codepoint=122), min_size=1).map(lambda s: '_'.join(s[i:i+1] for i in range(len(s)))))
@example('A_A')
@example('a_b')
@settings(max_examples=100)
def test_to_pascal_idempotent(field_name):
    """to_pascal applied twice should equal to_pascal applied once (idempotence)."""
    once = to_pascal(field_name)
    twice = to_pascal(once)
    assert once == twice, f"Failed for input '{field_name}': to_pascal('{field_name}')='{once}', to_pascal('{once}')='{twice}'"

try:
    test_to_pascal_idempotent()
    print("All tests passed!")
except AssertionError as e:
    print(f"Test failed: {e}")