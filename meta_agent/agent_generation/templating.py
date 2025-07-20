from .schemas import AgentBlueprint

def create_agent_files(agent_blueprint: AgentBlueprint):
    """
    Creates the agent.py and __init__.py files from an agent blueprint.
    """
    agent_py_template = """\
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from typing import Dict, Any

from google.adk.agents import Agent
from google.adk.tools.google_search import GoogleSearch
from google.adk.tools.code_executor import BuiltInCodeExecutor
import google.auth.transport.requests
import google.oauth2.id_token

# Initialize Opik for monitoring
try:
    from opik import Opik
    opik_client = Opik()
    HAS_OPIK = True
except ImportError:
    HAS_OPIK = False


class {agent_class_name}(Agent):
    \"\"\"
    {description}
    \"\"\"
    
    def __init__(self):
        super().__init__()
        self.instructions = \"\"\"{instructions}\"\"\"
        
        # Initialize tools based on agent blueprint
        self.tools = {{}}
        {tool_initialization}
        
    def run(self, user_input: str, **kwargs) -> str:
        \"\"\"
        Process user input and generate a response using available tools.
        
        Args:
            user_input: The user's message or query
            **kwargs: Additional parameters
            
        Returns:
            The agent's response
        \"\"\"
        if HAS_OPIK:
            with opik_client.trace(name="{agent_name}_run") as trace:
                trace.update(input={{"user_input": user_input}})
                response = self._process_request(user_input, **kwargs)
                trace.update(output={{"response": response}})
                return response
        else:
            return self._process_request(user_input, **kwargs)
    
    def _process_request(self, user_input: str, **kwargs) -> str:
        \"\"\"
        Internal method to process the request.
        \"\"\"
        try:
            # Basic agent logic - this would be enhanced based on specific requirements
            context = f"Instructions: {{self.instructions}}\\n\\nUser Query: {{user_input}}"
            
            # For now, return a simple response
            # In a real implementation, this would use the LLM and tools
            return f"I am {{self.__class__.__name__}} and I received: {{user_input}}"
            
        except Exception as e:
            return f"Error processing request: {{str(e)}}"


# Export the agent
agent = {agent_class_name}()
"""

    init_py_template = """\
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .agent import {agent_class_name}, agent

__all__ = ["{agent_class_name}", "agent"]
"""

    server_py_template = """\
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import agent

app = FastAPI(title="{agent_name}", description="{description}")

class QueryRequest(BaseModel):
    message: str
    
class QueryResponse(BaseModel):
    response: str

@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    \"\"\"Query the agent with a message.\"\"\"
    try:
        response = agent.run(request.message)
        return QueryResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    \"\"\"Health check endpoint.\"\"\"
    return {{"status": "healthy", "agent": "{agent_name}"}}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
"""

    # Generate tool initialization code
    tool_init_lines = []
    for tool in agent_blueprint.tools:
        if tool == "google_search":
            tool_init_lines.append('        self.tools["google_search"] = GoogleSearch()')
        elif tool == "code_executor":
            tool_init_lines.append('        self.tools["code_executor"] = BuiltInCodeExecutor()')
        else:
            # Generic tool placeholder
            tool_init_lines.append(f'        # TODO: Initialize {tool} tool')
    
    tool_initialization = "\n".join(tool_init_lines) if tool_init_lines else "        # No tools configured"

    # Convert agent name to proper class name
    agent_class_name = "".join(word.capitalize() for word in agent_blueprint.agent_name.split('_'))

    agent_py_content = agent_py_template.format(
        agent_class_name=agent_class_name,
        agent_name=agent_blueprint.agent_name,
        description=agent_blueprint.description,
        instructions=agent_blueprint.instructions,
        tool_initialization=tool_initialization
    )

    init_py_content = init_py_template.format(
        agent_class_name=agent_class_name
    )
    
    server_py_content = server_py_template.format(
        agent_name=agent_blueprint.agent_name,
        description=agent_blueprint.description
    )

    return {
        "agent.py": agent_py_content,
        "__init__.py": init_py_content,
        "server.py": server_py_content
    }
