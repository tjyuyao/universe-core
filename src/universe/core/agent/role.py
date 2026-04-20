from pydantic import BaseModel, Field
from .mindset import Mindset

class Role(BaseModel):
    name: str
    description: str
    mindsets: dict[str, Mindset] = Field(default_factory=dict)

    def add_mindset(self, mindset: Mindset):
        self.mindsets[mindset.name] = mindset
    
    def get_mindset(self, name: str) -> Mindset:
        return self.mindsets[name]
    
    def remove_mindset(self, name: str) -> Mindset:
        return self.mindsets.pop(name)