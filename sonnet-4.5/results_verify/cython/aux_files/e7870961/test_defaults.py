import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita._tempita import parse_signature

# Test with default values
result = parse_signature("name='default'", "test", (1, 1))
sig_args, _, _, _ = result
print(f"Signature \"name='default'\" parsed as: {sig_args}")

result2 = parse_signature("name, greeting='hello'", "test", (1, 1))
sig_args2, _, _, _ = result2
print(f"Signature \"name, greeting='hello'\" parsed as: {sig_args2}")

# Test tokenizer output for debugging
import tokenize
import io

def debug_tokens(sig_text):
    print(f"\nTokens for '{sig_text}':")
    tokens = tokenize.generate_tokens(io.StringIO(sig_text).readline)
    for tok_type, tok_string, _, _, _ in tokens:
        tok_name = tokenize.tok_name[tok_type]
        print(f"  {tok_name}('{tok_string}')")

debug_tokens("name")
debug_tokens("name, greeting")