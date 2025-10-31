# Testing the logic flow of the read_bytes function

sample_values = [False, 0, 1, 2, "0", "1 B"]

for sample in sample_values:
    print(f"\nsample={sample!r} (type: {type(sample).__name__})")
    print(f"  if sample: → {bool(sample)}")

    # This is the actual check in the code (line 163)
    if sample:
        print("  Would enter the sample processing block")
        if sample is True:
            print("    sample would be set to '10 kiB'")
        elif isinstance(sample, str):
            print(f"    sample would be parsed as bytes: parse_bytes('{sample}')")
        else:
            print(f"    sample would remain as: {sample}")
    else:
        print("  Would skip sample processing, returning the original value")