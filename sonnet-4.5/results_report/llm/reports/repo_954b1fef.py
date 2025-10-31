from llm.utils import truncate_string

text = "Hello world"

for max_length in [0, 1, 2, 3, 4]:
    result = truncate_string(text, max_length=max_length)
    print(f"max_length={max_length}: '{result}' (actual length={len(result)})")