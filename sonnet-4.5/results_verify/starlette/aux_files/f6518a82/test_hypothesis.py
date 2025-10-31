from hypothesis import given, strategies as st, settings, assume
import numpy as np
import numpy.strings as nps

@given(st.text(min_size=1, max_size=20).filter(lambda x: '\x00' not in x),
       st.text(min_size=1, max_size=20).filter(lambda x: '\x00' not in x),
       st.text(max_size=10).filter(lambda x: '\x00' not in x))
@settings(max_examples=500)
def test_replace_matches_python(prefix, old, suffix):
    assume(len(old) > 0)
    assume(len(prefix) + len(old) + len(suffix) < 50)

    s = prefix + old + suffix
    new = 'X' * (len(old) + 5) if len(old) < 10 else 'Y'

    arr = np.array([s])
    result = nps.replace(arr, old, new, count=1)
    python_result = s.replace(old, new, 1)

    assert result[0] == python_result, f"Failed: s={s!r}, old={old!r}, new={new!r}, numpy={result[0]!r}, python={python_result!r}"

# Run the test
try:
    test_replace_matches_python()
    print("All tests passed!")
except Exception as e:
    print(f"Test failed: {e}")