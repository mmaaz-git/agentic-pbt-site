import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from Cython.Compiler.StringEncoding import split_string_literal


@given(
    st.integers(min_value=200, max_value=500),
    st.integers(min_value=5, max_value=15)
)
@settings(max_examples=50, deadline=5000)
def test_split_string_literal_backslash_performance(num_backslashes, limit):
    s = '\\' * num_backslashes
    result = split_string_literal(s, limit=limit)
    rejoined = result.replace('""', '')
    assert rejoined == s
    print(f"Test passed for {num_backslashes} backslashes with limit {limit}")

if __name__ == "__main__":
    test_split_string_literal_backslash_performance()