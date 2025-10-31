import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from Cython.Tempita._tempita import parse_signature

@given(st.lists(st.text(alphabet='abcdefg', min_size=1, max_size=5).filter(str.isidentifier),
                min_size=1, max_size=3, unique=True))
def test_parse_signature_preserves_all_arguments(arg_names):
    sig_text = ', '.join(arg_names)
    sig_args, _, _, _ = parse_signature(sig_text, "test", (1, 1))

    assert len(sig_args) == len(arg_names), \
        f"Expected {len(arg_names)} args, got {len(sig_args)}"

    for name in arg_names:
        assert name in sig_args

# Run the test
if __name__ == "__main__":
    test_parse_signature_preserves_all_arguments()