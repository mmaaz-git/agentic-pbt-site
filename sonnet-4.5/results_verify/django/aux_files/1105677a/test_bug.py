from hypothesis import given, strategies as st, settings, example
import string


def to_camel_case(name):
    """Convert name to camel case as done in Django templates.py line 138."""
    return "".join(x for x in name.title() if x != "_")


@given(st.text(alphabet=string.ascii_lowercase + string.digits, min_size=3, max_size=20))
@example("my2app")
@example("app2api")
@example("test1module")
@settings(max_examples=500)
def test_camel_case_unexpected_capitals_after_digits(name):
    result = to_camel_case(name)

    for i in range(len(name) - 1):
        if name[i].isdigit() and name[i+1].isalpha():
            title_name = name.title()
            is_uppercase_after_digit = title_name[i+1].isupper()

            if is_uppercase_after_digit and "_" not in name[:i+2]:
                assert False, f"Bug: '{name}' -> '{result}' - letter after digit unexpectedly capitalized"


if __name__ == "__main__":
    # Run the test
    try:
        test_camel_case_unexpected_capitals_after_digits()
        print("All tests passed")
    except AssertionError as e:
        print(f"Test failed: {e}")