import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import pandas as pd
from hypothesis import given, strategies as st, settings


@given(st.text(max_size=50), st.integers(min_value=-10, max_value=10), st.integers(min_value=-10, max_value=10), st.text(max_size=20))
@settings(max_examples=1000)
def test_slice_replace_matches_python(text, start, stop, repl):
    s = pd.Series([text])
    pandas_result = s.str.slice_replace(start, stop, repl)[0]
    python_slice = text[:start] + repl + text[stop:]
    assert pandas_result == python_slice, f"text={repr(text)}, start={start}, stop={stop}, repl={repr(repl)}: pandas={repr(pandas_result)}, python={repr(python_slice)}"

# Run the test
test_slice_replace_matches_python()