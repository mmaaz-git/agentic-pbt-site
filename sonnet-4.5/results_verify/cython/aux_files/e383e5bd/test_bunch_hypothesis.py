import keyword
from hypothesis import given, strategies as st
import Cython.Tempita as tempita

RESERVED = {"if", "for", "endif", "endfor", "else", "elif", "py", "default", "inherit"} | set(keyword.kwlist)
valid_identifier = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122),
    min_size=1,
    max_size=10
).filter(lambda s: s not in RESERVED and s.isidentifier())


@given(st.dictionaries(valid_identifier, st.integers(), min_size=1, max_size=5))
def test_bunch_dir_missing_attrs(kwargs):
    b = tempita.bunch(**kwargs)

    for key in kwargs:
        assert hasattr(b, key), f"hasattr failed for key {key}"
        assert key not in dir(b), f"key {key} found in dir(b) when it shouldn't be"

if __name__ == "__main__":
    # Test with the specific failing input
    kwargs = {'a': 0}
    b = tempita.bunch(**kwargs)

    print(f"Testing with kwargs={kwargs}")
    print(f"b.a = {b.a}")
    print(f"hasattr(b, 'a') = {hasattr(b, 'a')}")
    print(f"'a' in dir(b) = {'a' in dir(b)}")

    # Run the property-based test
    test_bunch_dir_missing_attrs()
    print("\nHypothesis test passed - confirming the bug exists")