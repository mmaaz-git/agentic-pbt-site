"""Test file to reproduce the _check_for_default_values bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.util._validators import _check_for_default_values


class UncomparableObject:
    def __eq__(self, other):
        raise TypeError("Cannot compare UncomparableObject")


class RaisesAttributeError:
    def __eq__(self, other):
        raise AttributeError("No such attribute")


def test_typeerror():
    """Test that TypeError propagates instead of being caught."""
    obj = UncomparableObject()
    arg_val_dict = {'param': obj}
    compat_args = {'param': obj}  # Same object, so 'is' would return True

    try:
        _check_for_default_values('test_func', arg_val_dict, compat_args)
        print("TEST 1: No exception raised - unexpected!")
        return False
    except TypeError as e:
        print(f"TEST 1: TypeError raised as expected by bug report: {e}")
        return True
    except ValueError as e:
        print(f"TEST 1: ValueError raised: {e}")
        return False
    except Exception as e:
        print(f"TEST 1: Different exception raised: {type(e).__name__}: {e}")
        return False


def test_attributeerror():
    """Test that AttributeError also propagates."""
    obj = RaisesAttributeError()
    arg_val_dict = {'param': obj}
    compat_args = {'param': obj}  # Same object, so 'is' would return True

    try:
        _check_for_default_values('test_func', arg_val_dict, compat_args)
        print("TEST 2: No exception raised - unexpected!")
        return False
    except AttributeError as e:
        print(f"TEST 2: AttributeError raised as expected: {e}")
        return True
    except ValueError as e:
        print(f"TEST 2: ValueError raised: {e}")
        return False
    except Exception as e:
        print(f"TEST 2: Different exception raised: {type(e).__name__}: {e}")
        return False


def test_valueerror_is_caught():
    """Test that ValueError IS currently caught (the existing behavior)."""
    class RaisesValueError:
        def __eq__(self, other):
            raise ValueError("Cannot compare - ValueError")

    obj = RaisesValueError()
    arg_val_dict = {'param': obj}
    compat_args = {'param': obj}  # Same object, so 'is' would return True

    try:
        _check_for_default_values('test_func', arg_val_dict, compat_args)
        print("TEST 3: No exception raised - ValueError was caught and 'is' comparison succeeded!")
        return True
    except ValueError as e:
        print(f"TEST 3: ValueError propagated: {e}")
        return False
    except Exception as e:
        print(f"TEST 3: Different exception raised: {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("REPRODUCING THE BUG")
    print("=" * 60)

    bug1 = test_typeerror()
    bug2 = test_attributeerror()
    correct = test_valueerror_is_caught()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if bug1 and bug2:
        print("BUG CONFIRMED: TypeError and AttributeError propagate instead of being caught.")
        print("Only ValueError is currently caught for fallback to 'is' comparison.")
    elif correct:
        print("ValueError IS correctly caught and falls back to 'is' comparison.")
    else:
        print("Unexpected behavior detected.")