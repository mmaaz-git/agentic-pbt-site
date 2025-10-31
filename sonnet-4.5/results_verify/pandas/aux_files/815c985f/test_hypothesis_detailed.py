from hypothesis import given, strategies as st
import inspect


def test_platform_power_docstring_consistency():
    from pandas.compat import is_platform_power

    source = inspect.getsource(is_platform_power)
    doc = is_platform_power.__doc__

    print("Source code check: 'ppc64' in source:", "ppc64" in source)
    print("Docstring check: 'ARM' not in doc:", "ARM" not in doc)
    print("Docstring check: 'Power' in doc:", "Power" in doc)
    print("\nDocstring content:")
    print(doc)

    # The actual assertion from the bug report
    assert "ppc64" in source
    assert "ARM" not in doc or "Power" in doc

    # Let's check what actually fails
    if "ARM" in doc and "Power" not in doc:
        print("\nBUG CONFIRMED: Docstring incorrectly mentions ARM instead of Power")
        return False
    return True

if __name__ == "__main__":
    result = test_platform_power_docstring_consistency()
    if not result:
        print("Test revealed the bug")