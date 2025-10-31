from pydantic import BaseModel, Field

# Test case 1: With default value
class ExcludeModel(BaseModel):
    public: int
    private: int = Field(default=0, exclude=True)

model = ExcludeModel(public=1, private=2)
print(f"Original: public={model.public}, private={model.private}")

dumped = model.model_dump()
print(f"Dumped: {dumped}")

restored = ExcludeModel.model_validate(dumped)
print(f"Restored: public={restored.public}, private={restored.private}")

print(f"Are they equal? {model == restored}")
print(f"Private field preserved? {model.private == restored.private}")

print("\n" + "="*50 + "\n")

# Test case 2: Without default value (should raise ValidationError)
class RequiredExcludeModel(BaseModel):
    public: int
    private: int = Field(exclude=True)

model2 = RequiredExcludeModel(public=1, private=2)
print(f"Original model2: public={model2.public}, private={model2.private}")

dumped2 = model2.model_dump()
print(f"Dumped2: {dumped2}")

try:
    restored2 = RequiredExcludeModel.model_validate(dumped2)
    print(f"Restored2: public={restored2.public}, private={restored2.private}")
except Exception as e:
    print(f"Error during validation: {type(e).__name__}: {e}")