import math

def _adjust_split_out_for_group_keys(npartitions, by):
    if len(by) == 1:
        return math.ceil(npartitions / 15)
    return math.ceil(npartitions / (10 / (len(by) - 1)))

# Test case that crashes
npartitions = 100
by = []

print("Testing _adjust_split_out_for_group_keys with empty by list:")
print(f"npartitions = {npartitions}")
print(f"by = {by}")
print(f"len(by) = {len(by)}")
print()

try:
    result = _adjust_split_out_for_group_keys(npartitions, by)
    print(f"Result: {result}")

    # Show the calculation step by step
    print("\nStep-by-step calculation:")
    print(f"len(by) - 1 = {len(by)} - 1 = {len(by) - 1}")
    denominator = 10 / (len(by) - 1)
    print(f"10 / (len(by) - 1) = 10 / {len(by) - 1} = {denominator}")
    print(f"npartitions / denominator = {npartitions} / {denominator} = {npartitions / denominator}")
    print(f"math.ceil({npartitions / denominator}) = {math.ceil(npartitions / denominator)}")

    # Check if the result makes sense
    print(f"\nIs result positive? {result > 0}")
    print(f"Is result a valid number of partitions? {result > 0 and isinstance(result, int)}")

except Exception as e:
    print(f"Error: {e}")