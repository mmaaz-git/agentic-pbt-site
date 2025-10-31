from hypothesis import given, strategies as st, settings, example
from django.db.backends.utils import truncate_name


def calculate_identifier_length(identifier):
    stripped = identifier.strip('"')
    return len(stripped)


@given(
    namespace=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu',))),
    table_name=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=('Lu',))),
    length=st.integers(min_value=10, max_value=30)
)
@settings(max_examples=1000)
@example(namespace='SCHEMA', table_name='VERYLONGTABLENAME', length=20)
def test_truncate_name_respects_length_with_namespace(namespace, table_name, length):
    identifier = f'{namespace}"."{table_name}'
    result = truncate_name(identifier, length=length)
    result_length = calculate_identifier_length(result)

    assert result_length <= length


if __name__ == "__main__":
    test_truncate_name_respects_length_with_namespace()