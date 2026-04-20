from pydantic import BaseModel, Field
from ..object_ import Channel


class Mindset(BaseModel):
    name: str
    description: str
    channels: dict[str, Channel] = Field(default_factory=dict)  # cognitive target -> channel
    model: str | None = None
    
    def add_channel(self, channel: Channel):
        self.channels[channel.cognitive_target] = channel

    def get_channel(self, cognitive_target: str) -> Channel:
        return self.channels[cognitive_target]
    
    def remove_channel(self, cognitive_target: str) -> Channel:
        return self.channels.pop(cognitive_target)
