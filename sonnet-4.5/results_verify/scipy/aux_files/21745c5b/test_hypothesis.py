import pandas as pd
import inspect

def test_split_rsplit_have_same_parameters():
    split_sig = inspect.signature(pd.core.strings.accessor.StringMethods.split)
    rsplit_sig = inspect.signature(pd.core.strings.accessor.StringMethods.rsplit)

    split_params = set(split_sig.parameters.keys())
    rsplit_params = set(rsplit_sig.parameters.keys())

    print(f"split() parameters: {split_params}")
    print(f"rsplit() parameters: {rsplit_params}")

    print(f"\n'regex' in split_params: {'regex' in split_params}")
    print(f"'regex' in rsplit_params: {'regex' in rsplit_params}")

    assert 'regex' in split_params
    assert 'regex' in rsplit_params

if __name__ == "__main__":
    try:
        test_split_rsplit_have_same_parameters()
        print("\nTest PASSED")
    except AssertionError as e:
        print(f"\nTest FAILED with assertion error")