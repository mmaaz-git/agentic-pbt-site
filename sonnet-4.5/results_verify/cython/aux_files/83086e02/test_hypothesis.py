import os
import tempfile
from hypothesis import given, settings, strategies as st
from Cython.Build.Dependencies import extended_iglob

@given(st.lists(st.text(alphabet=st.characters(min_codepoint=97, max_codepoint=122), min_size=1, max_size=10), min_size=1, max_size=5))
@settings(max_examples=200)
def test_extended_iglob_no_duplicates(alternatives):
    with tempfile.TemporaryDirectory() as tmpdir:
        for alt in alternatives:
            filepath = os.path.join(tmpdir, f"{alt}.txt")
            with open(filepath, 'w') as f:
                f.write('')

        pattern = os.path.join(tmpdir, '{' + ','.join(alternatives) + '}.txt')
        results = list(extended_iglob(pattern))

        assert len(results) == len(set(results)), \
            f"Found duplicates in extended_iglob results: {results}"

if __name__ == '__main__':
    # Run the test
    test_extended_iglob_no_duplicates()