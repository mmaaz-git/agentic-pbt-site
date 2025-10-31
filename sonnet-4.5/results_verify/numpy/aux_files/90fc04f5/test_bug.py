import pandas.tseries.frequencies

print("Testing the bug report examples:")
print("=" * 50)

examples = ['MS', 'QS', 'BQE', 'BQS']

for offset_str in examples:
    first = pandas.tseries.frequencies.get_period_alias(offset_str)
    second = pandas.tseries.frequencies.get_period_alias(first) if first else None
    print(f"get_period_alias('{offset_str}') = {first}")
    print(f"get_period_alias({first!r}) = {second}")
    if second != first:
        print(f"  ⚠️ Idempotence violated!")
    else:
        print(f"  ✓ Idempotent")
    print()

print("\nAdditional test - checking if M and Q themselves map to anything:")
print("=" * 50)
print(f"get_period_alias('M') = {pandas.tseries.frequencies.get_period_alias('M')}")
print(f"get_period_alias('Q') = {pandas.tseries.frequencies.get_period_alias('Q')}")