import pandas as pd
import inspect

# Property-based test
def test_split_rsplit_have_same_parameters():
    split_sig = inspect.signature(pd.core.strings.accessor.StringMethods.split)
    rsplit_sig = inspect.signature(pd.core.strings.accessor.StringMethods.rsplit)

    split_params = set(split_sig.parameters.keys())
    rsplit_params = set(rsplit_sig.parameters.keys())

    print("split() parameters:", split_params)
    print("rsplit() parameters:", rsplit_params)
    print("split() has 'regex':", 'regex' in split_params)
    print("rsplit() has 'regex':", 'regex' in rsplit_params)

    assert 'regex' in split_params
    assert 'regex' in rsplit_params

# Run the test
try:
    test_split_rsplit_have_same_parameters()
    print("Test passed")
except AssertionError as e:
    print("Test failed - rsplit() does not have 'regex' parameter")

print("\n" + "="*50 + "\n")

# Reproduction code
s = pd.Series(['a.b.c.d'])

print("s.str.split('.', regex=True).iloc[0]:")
print(s.str.split('.', regex=True).iloc[0])
print()

print("s.str.split('.', regex=False).iloc[0]:")
print(s.str.split('.', regex=False).iloc[0])
print()

print("s.str.rsplit('.').iloc[0]:")
print(s.str.rsplit('.').iloc[0])
print()

print("Trying s.str.rsplit('.', regex=False):")
try:
    s.str.rsplit('.', regex=False)
    print("Success")
except TypeError as e:
    print(f"ERROR: {e}")