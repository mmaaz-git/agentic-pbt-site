import pandas.io.clipboard as clipboard

original_paste = clipboard.paste
call_count = [0]

def mock_paste():
    call_count[0] += 1
    if call_count[0] == 1:
        return "initial text"
    else:
        return None

clipboard.paste = mock_paste

result = clipboard.waitForNewPaste(timeout=0.1)
print(f"Result: {repr(result)}, Type: {type(result).__name__}")

clipboard.paste = original_paste