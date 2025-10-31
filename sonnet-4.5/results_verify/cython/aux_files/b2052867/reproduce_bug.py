import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Compiler.PyrexTypes import cap_length

result = cap_length('00', 1)
print(f"cap_length('00', max_len=1) = {repr(result)}")
print(f"Length: {len(result)} (expected: <= 1)")

result2 = cap_length('x' * 100, 10)
print(f"cap_length('x'*100, max_len=10) = {repr(result2)}")
print(f"Length: {len(result2)} (expected: <= 10)")

# Test more edge cases
result3 = cap_length('abc', 2)
print(f"\ncap_length('abc', max_len=2) = {repr(result3)}")
print(f"Length: {len(result3)} (expected: <= 2)")

result4 = cap_length('test', 16)
print(f"\ncap_length('test', max_len=16) = {repr(result4)}")
print(f"Length: {len(result4)} (expected: <= 16)")

result5 = cap_length('a' * 50, 17)
print(f"\ncap_length('a'*50, max_len=17) = {repr(result5)}")
print(f"Length: {len(result5)} (expected: <= 17)")