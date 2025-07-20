from .schemas import AgentBlueprint

def create_agent_files(agent_blueprint: AgentBlueprint):
    """
    Creates the agent.py and __init__.py files from an agent blueprint.
    """
    agent_py_template = """\
from google.adk.tools import *
from google.adk.runtime import *

class {agent_name}(Agent):
    def __init__(self):
        super().__init__()
        self.instructions = \"\"\"{instructions}\"\"\"
        self.tools = [{tools}]

    def call(self, message):
        # Your agent logic here
        pass
"""

    init_py_template = """\
from .{agent_name} import {agent_name}
"""

    agent_py_content = agent_py_template.format(
        agent_name=agent_blueprint.agent_name,
        instructions=agent_blueprint.instructions,
        tools=", ".join(agent_blueprint.tools)
    )

    init_py_content = init_py_template.format(
        agent_name=agent_blueprint.agent_name
    )

    return {
        "agent.py": agent_py_content,
        "__init__.py": init_py_content
    }
