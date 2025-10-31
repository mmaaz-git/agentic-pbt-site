import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')
import operator

# Test operator.__not__
print("Testing operator.__not__:")
print(f"  operator.__not__ exists: {hasattr(operator, '__not__')}")
print(f"  operator.__not__(True) = {operator.__not__(True)}")
print(f"  operator.__not__(False) = {operator.__not__(False)}")

# Test the logic used in the constraint
values = [5, 10, 15]
test_values = [3, 5, 10, 20]

print("\nTesting constraint logic:")
for v in test_values:
    in_result = operator.__contains__(values, v)
    not_in_result = operator.__not__(operator.__contains__(values, v))
    print(f"  v={v}: in={in_result}, not_in={not_in_result}, expected_not_in={v not in values}")

# Verify they match
print("\nVerification:")
for v in test_values:
    actual = operator.__not__(operator.__contains__(values, v))
    expected = v not in values
    if actual != expected:
        print(f"  MISMATCH: v={v}, actual={actual}, expected={expected}")
    else:
        print(f"  OK: v={v}, result={actual}")