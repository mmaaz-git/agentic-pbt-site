import llm.utils

test_cases = [
    ("Hello, World!", 1),
    ("Hello, World!", 2),
    ("Hello, World!", 3),
    ("A", 1),
    ("AB", 1),
    ("ABC", 2),
]

for text, max_length in test_cases:
    result = llm.utils.truncate_string(text, max_length=max_length)
    violation = "❌ VIOLATION" if len(result) > max_length else "✓"
    print(f"text='{text[:20]}...' (len={len(text):2}), max_length={max_length:2} → '{result}' (len={len(result):2}) {violation}")