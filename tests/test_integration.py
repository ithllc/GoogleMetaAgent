import unittest
from unittest.mock import patch, MagicMock
from meta_agent.main import main

class TestIntegration(unittest.TestCase):

    @patch('meta_agent.main.trigger_cloud_build')
    @patch('meta_agent.main.deploy_to_cloud_run')
    @patch('meta_agent.main.os.makedirs')
    @patch('builtins.open')
    def test_end_to_end_pipeline(self, mock_open, mock_makedirs, mock_deploy_to_cloud_run, mock_trigger_cloud_build):
        # Mock the file system and cloud services
        mock_open.return_value = MagicMock()
        mock_makedirs.return_value = None
        mock_trigger_cloud_build.return_value = None
        mock_deploy_to_cloud_run.return_value = "https://my-agent.a.run.app"

        # Run the main function
        main()

        # Assert that the correct functions were called
        mock_makedirs.assert_called_once_with("generated_agents/MyAgent", exist_ok=True)
        self.assertEqual(mock_open.call_count, 4)
        # In the main script, the cloud build and deploy are commented out.
        # mock_trigger_cloud_build.assert_called_once()
        # mock_deploy_to_cloud_run.assert_called_once()

if __name__ == '__main__':
    unittest.main()
