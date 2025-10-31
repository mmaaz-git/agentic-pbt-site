import sys
import tempfile
import os

# Add scipy to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/scipy_env/lib/python3.13/site-packages')

from scipy.datasets._utils import _clear_cache

method_map = {"test": ["test.dat"]}

print("Testing assert behavior with non-callable input...")
print("-" * 50)

with tempfile.TemporaryDirectory() as tmpdir:
    test_file = os.path.join(tmpdir, "test.dat")
    with open(test_file, 'w') as f:
        f.write("test")

    non_callable = "not_a_function"

    print("Test 1: Running WITHOUT -O flag (assertions enabled)")
    try:
        _clear_cache([non_callable], cache_dir=tmpdir, method_map=method_map)
        print("No error raised - THIS IS UNEXPECTED!")
    except Exception as e:
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        import traceback
        print(f"Traceback line with error: {traceback.format_exc().splitlines()[-2]}")

    print()