from pydantic import BaseModel, Field

class ExcludeModel(BaseModel):
    public: int
    private: int = Field(default=0, exclude=True)

model = ExcludeModel(public=1, private=2)
print(f"Original: public={model.public}, private={model.private}")

# Try different approaches
print("\n1. Default model_dump():")
dumped1 = model.model_dump()
print(f"   Result: {dumped1}")

print("\n2. model_dump(exclude=set()):")
dumped2 = model.model_dump(exclude=set())
print(f"   Result: {dumped2}")

print("\n3. model_dump(include={'public', 'private'}):")
dumped3 = model.model_dump(include={'public', 'private'})
print(f"   Result: {dumped3}")

print("\n4. model_dump(mode='python'):")
dumped4 = model.model_dump(mode='python')
print(f"   Result: {dumped4}")

print("\n5. dict(model):")
dumped5 = dict(model)
print(f"   Result: {dumped5}")

# Check model_copy
print("\n6. model.model_copy():")
copied = model.model_copy()
print(f"   Copied: public={copied.public}, private={copied.private}")
print(f"   Private field preserved in copy? {model.private == copied.private}")