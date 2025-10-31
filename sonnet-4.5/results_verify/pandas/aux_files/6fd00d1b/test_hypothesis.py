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
test_options_reset()
print("Test passed!")