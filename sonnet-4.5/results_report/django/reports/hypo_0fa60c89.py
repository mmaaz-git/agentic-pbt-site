from hypothesis import given, strategies as st
from django.db.migrations.serializer import DictionarySerializer

@given(st.dictionaries(
    st.one_of(st.integers(), st.text()),
    st.one_of(st.integers(), st.text()),
    max_size=10
))
def test_dictionary_serializer_deterministic(d):
    serializer = DictionarySerializer(d)
    result, _ = serializer.serialize()

# Run the test
if __name__ == "__main__":
    test_dictionary_serializer_deterministic()