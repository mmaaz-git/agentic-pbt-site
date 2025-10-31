import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from Cython.Utils import build_hex_version

@given(st.integers(min_value=0, max_value=99),
       st.integers(min_value=0, max_value=99),
       st.integers(min_value=1, max_value=99))
def test_release_status_ordering_alpha_vs_final(major, minor, alpha_ver):
    version_alpha = f"{major}.{minor}a{alpha_ver}"
    version_final = f"{major}.{minor}"

    hex_alpha = build_hex_version(version_alpha)
    hex_final = build_hex_version(version_final)

    val_alpha = int(hex_alpha, 16)
    val_final = int(hex_final, 16)

    assert val_final > val_alpha, f"Final version {version_final} ({hex_final}) should be greater than alpha {version_alpha} ({hex_alpha})"

if __name__ == "__main__":
    test_release_status_ordering_alpha_vs_final()