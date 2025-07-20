import dspy
from .schemas import AgentBlueprint
from typing import List

class GenerateAgentBlueprint(dspy.Signature):
    """
    Generates an agent blueprint from a natural language description.
    """
    description = dspy.InputField(desc="A natural language description of the agent's purpose.")
    agent_name = dspy.OutputField(desc="A unique name for the agent (snake_case format).")
    agent_description = dspy.OutputField(desc="A brief description of what the agent does.")
    instructions = dspy.OutputField(desc="Detailed instructions for the agent to follow.")
    tools = dspy.OutputField(desc="Comma-separated list of tools the agent can use (e.g., google_search, code_executor).")

class AgentGenerator(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_agent_blueprint = dspy.ChainOfThought(GenerateAgentBlueprint)

    def forward(self, description: str) -> AgentBlueprint:
        """
        Generate an agent blueprint from a natural language description.
        
        Args:
            description: Natural language description of the agent's purpose
            
        Returns:
            AgentBlueprint: A structured blueprint for the agent
        """
        try:
            # Use DSPy to generate the blueprint
            result = self.generate_agent_blueprint(description=description)
            
            # Parse tools from comma-separated string to list
            tools_list = [tool.strip() for tool in result.tools.split(',') if tool.strip()]
            
            return AgentBlueprint(
                agent_name=result.agent_name.lower().replace(' ', '_').replace('-', '_'),
                description=result.agent_description,
                instructions=result.instructions,
                tools=tools_list
            )
        except Exception as e:
            # Fallback to a simple rule-based generation if DSPy fails
            return self._fallback_generation(description)
    
    def _fallback_generation(self, description: str) -> AgentBlueprint:
        """
        Fallback method for generating agent blueprint when DSPy fails.
        """
        # Simple rule-based agent generation
        agent_name = "generated_agent"
        tools = ["google_search"]  # Default tool
        
        # Detect common patterns in the description
        description_lower = description.lower()
        
        if any(keyword in description_lower for keyword in ["search", "find", "lookup", "query"]):
            tools.append("google_search")
        
        if any(keyword in description_lower for keyword in ["code", "program", "execute", "run", "script"]):
            tools.append("code_executor")
        
        if any(keyword in description_lower for keyword in ["math", "calculate", "solve", "equation"]):
            tools.extend(["code_executor", "calculator"])
        
        if any(keyword in description_lower for keyword in ["rag", "retrieve", "document", "knowledge"]):
            tools.extend(["google_search", "document_retriever"])
        
        # Remove duplicates while preserving order
        tools = list(dict.fromkeys(tools))
        
        return AgentBlueprint(
            agent_name=agent_name,
            description=f"An AI agent that {description}",
            instructions=f"You are an AI agent designed to {description}. Use the available tools to help users accomplish their tasks effectively.",
            tools=tools
        )
