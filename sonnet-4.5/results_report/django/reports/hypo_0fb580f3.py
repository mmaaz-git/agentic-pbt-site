from hypothesis import given, strategies as st, settings
from django.db.backends.utils import truncate_name, split_identifier

@given(st.text(min_size=1), st.integers(min_value=1, max_value=1000), st.integers(min_value=1, max_value=10))
@settings(max_examples=1000)
def test_truncate_name_length_invariant(identifier, length, hash_len):
    result = truncate_name(identifier, length=length, hash_len=hash_len)

    namespace, name = split_identifier(result)
    name_length = len(name)

    assert name_length <= length, f"Truncated name '{name}' has length {name_length} > {length}"

if __name__ == "__main__":
    test_truncate_name_length_invariant()