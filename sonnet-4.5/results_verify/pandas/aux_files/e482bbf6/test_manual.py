import pandas.io.clipboard as clipboard

# Test with None
print("Testing with None:")
original_paste = clipboard.paste
clipboard.paste = lambda: None

result = clipboard.waitForPaste(timeout=0.05)
print(f"Result: {repr(result)}, Type: {type(result).__name__}")

clipboard.paste = original_paste

# Test with other non-string values
print("\nTesting with 0:")
clipboard.paste = lambda: 0
result = clipboard.waitForPaste(timeout=0.05)
print(f"Result: {repr(result)}, Type: {type(result).__name__}")
clipboard.paste = original_paste

print("\nTesting with False:")
clipboard.paste = lambda: False
result = clipboard.waitForPaste(timeout=0.05)
print(f"Result: {repr(result)}, Type: {type(result).__name__}")
clipboard.paste = original_paste

print("\nTesting with []:")
clipboard.paste = lambda: []
result = clipboard.waitForPaste(timeout=0.05)
print(f"Result: {repr(result)}, Type: {type(result).__name__}")
clipboard.paste = original_paste

print("\nTesting with {}:")
clipboard.paste = lambda: {}
result = clipboard.waitForPaste(timeout=0.05)
print(f"Result: {repr(result)}, Type: {type(result).__name__}")
clipboard.paste = original_paste