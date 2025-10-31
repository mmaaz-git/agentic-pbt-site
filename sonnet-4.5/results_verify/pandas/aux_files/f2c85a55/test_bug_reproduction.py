import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.TestUtils import _parse_pattern

# Test the reported failure cases
test_cases = [
    "/start",
    "/",
    ":/end",
    ":/"
]

for test_case in test_cases:
    print(f"Testing: {repr(test_case)}")
    try:
        result = _parse_pattern(test_case)
        print(f"  Result: {result}")
    except ValueError as e:
        print(f"  ValueError: {e}")
    except Exception as e:
        print(f"  Other exception: {type(e).__name__}: {e}")
    print()