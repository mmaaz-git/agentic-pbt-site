from hypothesis import given, strategies as st, settings
import pandas.core.ops as ops

@settings(max_examples=200)
@given(
    op_name=st.sampled_from(['add', 'sub', 'mul', 'div', 'truediv', 'floordiv', 'mod', 'pow']),
    typ=st.sampled_from(['series', 'dataframe']),
)
def test_make_flex_doc_supports_all_flex_methods(op_name, typ):
    result = ops.make_flex_doc(op_name, typ)
    assert isinstance(result, str)
    assert len(result) > 0

# Run the test
test_make_flex_doc_supports_all_flex_methods()