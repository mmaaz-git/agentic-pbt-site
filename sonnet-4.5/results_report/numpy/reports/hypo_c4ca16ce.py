from hypothesis import given, strategies as st, settings
from scipy.io import arff
from io import StringIO


@settings(max_examples=100)
@given(
    attr_name=st.text(
        alphabet=st.characters(min_codepoint=97, max_codepoint=122),
        min_size=1,
        max_size=20
    )
)
def test_quote_stripping_consistency(attr_name):
    single_quote_arff = f"""
@relation test
@attribute '{attr_name}' numeric
@data
1.0
"""

    double_quote_arff = f"""
@relation test
@attribute "{attr_name}" numeric
@data
1.0
"""

    f1 = StringIO(single_quote_arff)
    data1, meta1 = arff.loadarff(f1)

    f2 = StringIO(double_quote_arff)
    data2, meta2 = arff.loadarff(f2)

    name_with_single_quotes = meta1.names()[0]
    name_with_double_quotes = meta2.names()[0]

    assert name_with_single_quotes == attr_name
    assert name_with_double_quotes == attr_name
    assert name_with_single_quotes == name_with_double_quotes

# Run the test
test_quote_stripping_consistency()