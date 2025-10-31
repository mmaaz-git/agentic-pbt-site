import pandas as pd
import pandas.io.json as pj
from hypothesis import given, assume, settings, strategies as st

@given(st.data())
@settings(max_examples=50)
def test_build_table_schema_primary_key_type(data):
    # Generate a random DataFrame with dictionary data
    df_dict = data.draw(st.dictionaries(
        st.text(min_size=1, max_size=10, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
        st.lists(st.integers(), min_size=1, max_size=5),
        min_size=1,
        max_size=5
    ))

    # Ensure all lists have the same length (required for DataFrame)
    assume(all(len(v) == len(list(df_dict.values())[0]) for v in df_dict.values()))

    df = pd.DataFrame(df_dict)

    # Test with different primary_key values
    for pk_value in [None, True, False]:
        schema = pj.build_table_schema(df, index=True, primary_key=pk_value)

        # According to Table Schema spec, primaryKey should be either:
        # - a string (single field name)
        # - an array of strings (multiple field names)
        # - not present
        if 'primaryKey' in schema:
            pk = schema['primaryKey']
            assert isinstance(pk, (list, type(None))), \
                f"primaryKey should be list or None per Table Schema spec, got {type(pk)} with value {pk}"

# Run the test
if __name__ == "__main__":
    test_build_table_schema_primary_key_type()