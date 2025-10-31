import pandas.io.clipboard as clipboard

# Save the original paste function
original_paste = clipboard.paste

# Create a mock counter to track calls
call_count = [0]

def mock_paste():
    """Mock paste function that returns 'initial text' on first call, None on second"""
    call_count[0] += 1
    if call_count[0] == 1:
        return "initial text"
    else:
        return None  # This should cause the bug

# Replace the paste function with our mock
clipboard.paste = mock_paste

# Call waitForNewPaste - it should wait for a new text string
# But instead it will return None when it sees the change
result = clipboard.waitForNewPaste(timeout=0.1)

print(f"Result: {repr(result)}")
print(f"Type: {type(result).__name__}")

# Restore original paste function
clipboard.paste = original_paste