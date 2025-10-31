from pydantic import BaseModel, Field

class ExcludeModel(BaseModel):
    public: int
    private: int = Field(default=0, exclude=True)

model = ExcludeModel(public=1, private=2)
print(f"Original: public={model.public}, private={model.private}")

# Test the suggested workaround
dumped = model.model_dump(exclude=None)
print(f"Dumped with exclude=None: {dumped}")

restored = ExcludeModel.model_validate(dumped)
print(f"Restored: public={restored.public}, private={restored.private}")

print(f"Are they equal? {model == restored}")
print(f"Private field preserved? {model.private == restored.private}")