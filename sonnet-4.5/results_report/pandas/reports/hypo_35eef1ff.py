from hypothesis import given, strategies as st
import pandas.plotting._misc as misc

def test_options_reset():
    opts = misc._Options()

    opts["custom.key"] = True
    opts["x_compat"] = True

    opts.reset()

    assert "custom.key" not in opts
    assert opts["x_compat"] == False

# Run the test
if __name__ == "__main__":
    print("Running property-based test...")
    try:
        test_options_reset()
        print("Test passed!")
    except AssertionError as e:
        print("Test failed!")
        print(f"AssertionError: The test failed because custom.key persists after reset()")

        # Show the actual state to help understand the failure
        opts = misc._Options()
        opts["custom.key"] = True
        opts["x_compat"] = True
        print(f"\nBefore reset(): {dict(opts)}")
        opts.reset()
        print(f"After reset(): {dict(opts)}")
        print(f"\nExpected: custom.key should be removed after reset()")
        print(f"Actual: custom.key = {opts.get('custom.key', 'NOT PRESENT')}")

        # Re-raise to show the traceback
        raise