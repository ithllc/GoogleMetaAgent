from pydantic import BaseModel, Field
from typing import List

class AgentBlueprint(BaseModel):
    """
    A blueprint for a new agent.
    """
    agent_name: str = Field(..., description="The name of the agent.")
    description: str = Field(..., description="A brief description of the agent's purpose.")
    instructions: str = Field(..., description="The instructions for the agent to follow.")
    tools: List[str] = Field(..., description="A list of tools the agent can use.")
