import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from hypothesis import given, strategies as st, assume, settings
import pandas.io.formats.css as css


@given(
    value=st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
    unit=st.sampled_from(['pt', 'px', 'em', 'rem'])
)
@settings(max_examples=500)
def test_css_size_to_pt_always_returns_pt(value, unit):
    resolver = css.CSSResolver()
    if value < 0:
        assume(False)
    input_str = f"{value}{unit}"
    result = resolver.size_to_pt(input_str)
    assert result.endswith('pt'), f"Result should end with 'pt': {result}"

if __name__ == "__main__":
    test_css_size_to_pt_always_returns_pt()