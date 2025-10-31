from scipy.io.arff._arffread import NominalAttribute

def test_nominal_attribute_empty_values_should_not_crash(name):
    """NominalAttribute should either accept empty values or raise a descriptive error"""
    if name.strip() == '':
        return

    try:
        attr = NominalAttribute(name, ())
        assert hasattr(attr, 'dtype')
        print(f"Success with name: {name!r}")
    except ValueError as e:
        assert 'empty' in str(e).lower()
        print(f"Failed with name: {name!r}, error: {e}")

# Run the test with a few examples
test_nominal_attribute_empty_values_should_not_crash("test")
test_nominal_attribute_empty_values_should_not_crash("attr1")
test_nominal_attribute_empty_values_should_not_crash("my_attribute")