from pydantic.alias_generators import to_pascal

field = 'A_A'
once = to_pascal(field)
twice = to_pascal(once)

print(f"to_pascal('{field}') = '{once}'")
print(f"to_pascal('{once}') = '{twice}'")
print(f"Expected: '{once}' == '{twice}'")
print(f"Actual: '{once}' {'==' if once == twice else '!='} '{twice}'")

# Test a few more cases
test_cases = ['A_A', 'a_b', 'AA', 'test_case', 'TestCase', 'ABC', 'a_B_c']
for test in test_cases:
    once = to_pascal(test)
    twice = to_pascal(once)
    print(f"\nInput: '{test}'")
    print(f"  First application: '{once}'")
    print(f"  Second application: '{twice}'")
    print(f"  Idempotent: {once == twice}")