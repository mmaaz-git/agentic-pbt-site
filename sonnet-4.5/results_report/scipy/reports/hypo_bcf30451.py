from io import StringIO

from hypothesis import given, settings, strategies as st
from scipy.io import arff


@given(
    has_trailing_newline=st.booleans(),
    num_rows=st.integers(min_value=1, max_value=10),
    values=st.data()
)
@settings(max_examples=200)
def test_relational_attribute_handles_trailing_newline(has_trailing_newline, num_rows, values):
    rows = []
    for _ in range(num_rows):
        val = values.draw(st.floats(allow_nan=False, allow_infinity=False, min_value=-1e10, max_value=1e10))
        rows.append(str(val))

    relational_data = '\\n'.join(rows)
    if has_trailing_newline:
        relational_data += '\\n'

    content = f"""@relation test
@attribute id numeric
@attribute bag relational
  @attribute val numeric
@end bag
@data
1,"{relational_data}"
"""

    f = StringIO(content)
    data, meta = arff.loadarff(f)

    assert len(data) == 1

if __name__ == "__main__":
    test_relational_attribute_handles_trailing_newline()