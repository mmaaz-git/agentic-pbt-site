import pandas.compat

def test_is_platform_power_documentation_matches_implementation():
    result = pandas.compat.is_platform_power()

    doc = pandas.compat.is_platform_power.__doc__
    assert "Power architecture" in doc

    if "ARM architecture" in doc:
        raise AssertionError(
            "Documentation for is_platform_power() incorrectly states it checks for "
            "ARM architecture, but the implementation checks for Power architecture (ppc64, ppc64le). "
            "This is a documentation bug."
        )

# Run the test
try:
    test_is_platform_power_documentation_matches_implementation()
    print("Test passed (no documentation bug detected)")
except AssertionError as e:
    print(f"Test failed with error: {e}")