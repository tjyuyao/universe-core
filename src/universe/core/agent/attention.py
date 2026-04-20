from pydantic import BaseModel, Field
from ..object_ import Generic, Object, Action, Channel
from .mindset import Mindset
from .role import Role
from .soul import Soul


class Attention(BaseModel):
    current_soul: str = Field(default="")
    current_role: str = Field(default="")
    current_mindset: str = Field(default="")
    
    souls: dict[str, Soul] = Field(default_factory=dict)

    def add_soul(self, soul: Soul):
        self.souls[soul.name] = soul
    
    def get_soul(self, name: str):
        return self.souls[name]
    
    def remove_soul(self, name: str) -> Soul:
        return self.souls.pop(name)

    def get_current_soul(self) -> Soul:
        return self.get_soul(self.current_soul)
    
    def get_current_role(self) -> Role:
        return self.get_current_soul().get_role(self.current_role)
    
    def get_current_mindset(self) -> Mindset:
        return self.get_current_role().get_mindset(self.current_mindset)

    def get_current_channels(self) -> dict[str, Channel]:
        return self.get_current_mindset().channels
    
    def get_current_model_name(self) -> str | None:
        return self.get_current_mindset().model