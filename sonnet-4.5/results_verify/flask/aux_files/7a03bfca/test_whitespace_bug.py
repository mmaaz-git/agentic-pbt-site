"""Test to reproduce the Flask environment variable whitespace handling bug"""

import os
from hypothesis import given, strategies as st, example
from flask.helpers import get_debug_flag, get_load_dotenv

# First, let's run the Hypothesis test
@given(
    st.sampled_from(["0", "false", "no"]),
    st.text(alphabet=" \t", min_size=1, max_size=5),
    st.text(alphabet=" \t", min_size=1, max_size=5)
)
@example("false", " ", " ")
def test_get_debug_flag_should_strip_whitespace(falsy_value, prefix, suffix):
    value_with_whitespace = prefix + falsy_value + suffix

    original_val = os.environ.get("FLASK_DEBUG")
    try:
        os.environ["FLASK_DEBUG"] = value_with_whitespace
        result = get_debug_flag()

        assert result is False, (
            f"FLASK_DEBUG={value_with_whitespace!r} should disable debug mode, "
            f"but got {result}"
        )
    finally:
        if original_val is None:
            os.environ.pop("FLASK_DEBUG", None)
        else:
            os.environ["FLASK_DEBUG"] = original_val

# Run the hypothesis test
print("Running Hypothesis test...")
try:
    test_get_debug_flag_should_strip_whitespace()
    print("Hypothesis test passed (no bug found)")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")

# Now run the manual reproduction examples
print("\n=== Manual Reproduction Tests ===\n")

# Test get_debug_flag with whitespace
print("Testing get_debug_flag():")
os.environ["FLASK_DEBUG"] = " false "
result = get_debug_flag()
print(f"FLASK_DEBUG=' false ' returns: {result}")
assert result is True, "Expected True for ' false ' (bug exists)"

os.environ["FLASK_DEBUG"] = " 0 "
result = get_debug_flag()
print(f"FLASK_DEBUG=' 0 ' returns: {result}")
assert result is True, "Expected True for ' 0 ' (bug exists)"

os.environ["FLASK_DEBUG"] = " no "
result = get_debug_flag()
print(f"FLASK_DEBUG=' no ' returns: {result}")
assert result is True, "Expected True for ' no ' (bug exists)"

# Test get_load_dotenv with whitespace
print("\nTesting get_load_dotenv():")
os.environ["FLASK_SKIP_DOTENV"] = " false "
result = get_load_dotenv()
print(f"FLASK_SKIP_DOTENV=' false ' returns: {result}")
assert result is False, "Expected False for ' false ' (bug exists)"

os.environ["FLASK_SKIP_DOTENV"] = " 0 "
result = get_load_dotenv()
print(f"FLASK_SKIP_DOTENV=' 0 ' returns: {result}")
assert result is False, "Expected False for ' 0 ' (bug exists)"

os.environ["FLASK_SKIP_DOTENV"] = " no "
result = get_load_dotenv()
print(f"FLASK_SKIP_DOTENV=' no ' returns: {result}")
assert result is False, "Expected False for ' no ' (bug exists)"

# Test without whitespace (should work correctly)
print("\nTesting without whitespace (should work correctly):")
os.environ["FLASK_DEBUG"] = "false"
result = get_debug_flag()
print(f"FLASK_DEBUG='false' returns: {result}")
assert result is False, "Expected False for 'false'"

os.environ["FLASK_SKIP_DOTENV"] = "false"
result = get_load_dotenv()
print(f"FLASK_SKIP_DOTENV='false' returns: {result}")
assert result is True, "Expected True for 'false'"

print("\nAll tests completed successfully - bug is confirmed!")