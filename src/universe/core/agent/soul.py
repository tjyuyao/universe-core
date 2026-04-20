from pydantic import BaseModel, Field
from .role import Role


class Soul(BaseModel):
    name: str
    description: str
    roles: dict[str, Role] = Field(default_factory=dict)
    
    def add_role(self, role: Role):
        self.roles[role.name] = role
        
    def get_role(self, name: str):
        return self.roles[name]
    
    def remove_role(self, name: str) -> Role:
        return self.roles.pop(name)
    