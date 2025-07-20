import unittest
from meta_agent.agent_generation.generator import AgentGenerator
from meta_agent.agent_generation.schemas import AgentBlueprint
from meta_agent.agent_generation.templating import create_agent_files

class TestAgentGeneration(unittest.TestCase):

    def test_agent_generator(self):
        agent_generator = AgentGenerator()
        description = "A simple RAG agent that uses Google Search to answer questions."
        agent_blueprint = agent_generator.forward(description)
        self.assertIsInstance(agent_blueprint, AgentBlueprint)
        self.assertEqual(agent_blueprint.agent_name, "MyAgent")

    def test_create_agent_files(self):
        agent_blueprint = AgentBlueprint(
            agent_name="MyAgent",
            description="A simple agent",
            instructions="This is a simple agent that does nothing.",
            tools=["google_search"]
        )
        agent_files = create_agent_files(agent_blueprint)
        self.assertIn("agent.py", agent_files)
        self.assertIn("__init__.py", agent_files)
        self.assertIn("class MyAgent(Agent):", agent_files["agent.py"])

if __name__ == '__main__':
    unittest.main()
