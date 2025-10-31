from hypothesis import given, strategies as st
import inspect


def test_platform_power_docstring_consistency():
    from pandas.compat import is_platform_power

    source = inspect.getsource(is_platform_power)
    doc = is_platform_power.__doc__

    assert "ppc64" in source
    assert "ARM" not in doc or "Power" in doc

if __name__ == "__main__":
    test_platform_power_docstring_consistency()
    print("Test completed")