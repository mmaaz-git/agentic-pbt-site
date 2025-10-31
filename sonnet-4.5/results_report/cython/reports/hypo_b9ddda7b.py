import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, assume, settings
from Cython.Compiler.Code import _indent_chunk

@given(st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=0, max_size=200),
       st.integers(min_value=0, max_value=16))
@settings(max_examples=1000)
def test_indent_chunk_preserves_nonwhitespace(s, indent_len):
    assume('\t' not in s)
    result = _indent_chunk(s, indent_len)
    original_chars = ''.join(s.split())
    result_chars = ''.join(result.split())
    assert original_chars == result_chars, f"Content was lost: original={repr(s)}, result={repr(result)}"

if __name__ == "__main__":
    test_indent_chunk_preserves_nonwhitespace()