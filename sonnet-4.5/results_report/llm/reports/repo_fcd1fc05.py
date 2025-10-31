import sys
sys.path.insert(0, "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages")

from llm.utils import truncate_string

# Test the specific failing case from the bug report
result = truncate_string('00', max_length=1)
print(f"Input: text='00', max_length=1")
print(f"Expected: length <= 1")
print(f"Result: '{result}'")
print(f"Actual length: {len(result)}")
print(f"Violates contract: {len(result) > 1}")
print()

# Test a few more edge cases
test_cases = [
    ('abc', 2),
    ('hello', 3),
    ('test', 0),
    ('', 1),
    ('x', 1),
    ('xy', 1),
]

print("Additional test cases:")
for text, max_length in test_cases:
    result = truncate_string(text, max_length=max_length)
    violation = len(result) > max_length
    print(f"  text='{text}', max_length={max_length} -> '{result}' (len={len(result)}) {'VIOLATES' if violation else 'OK'}")