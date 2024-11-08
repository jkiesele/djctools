import unittest
import os
from djctools.wandb_tools import wandb_wrapper
import torch

class TestWandbWrapper(unittest.TestCase):

    def setUp(self):
        """Reset the singleton instance state before each test."""
        wandb_wrapper.log_buffer.clear()
        wandb_wrapper.active = True
        wandb_wrapper.initialized = False
        if "WANDB_API_KEY" in os.environ:
            del os.environ["WANDB_API_KEY"]

    def test_singleton_behavior(self):
        """Test that wandb_wrapper is a singleton instance."""
        instance1 = wandb_wrapper
        instance2 = wandb_wrapper
        self.assertIs(instance1, instance2, "wandb_wrapper should be a singleton")

    def test_activate_deactivate_logging(self):
        """Test activation and deactivation of logging."""
        wandb_wrapper.deactivate()
        self.assertFalse(wandb_wrapper.active, "Logging should be deactivated")
        wandb_wrapper.activate()
        self.assertTrue(wandb_wrapper.active, "Logging should be activated")

    def test_logging_buffering(self):
        """Test that metrics are buffered correctly when logging is active."""
        wandb_wrapper.log("accuracy", 0.95)
        self.assertIn("accuracy", wandb_wrapper.log_buffer)
        self.assertEqual(wandb_wrapper.log_buffer["accuracy"], 0.95)

    def test_logging_no_buffer_when_inactive(self):
        """Test that metrics are not buffered when logging is inactive."""
        wandb_wrapper.deactivate()
        wandb_wrapper.log("accuracy", 0.95)
        self.assertNotIn("accuracy", wandb_wrapper.log_buffer)

    def no_test_flush_clears_log_buffer(self):
        """Test that flush clears the log buffer."""
        wandb_wrapper.log("accuracy", 0.95)
        wandb_wrapper.flush()
        self.assertEqual(len(wandb_wrapper.log_buffer), 0, "Log buffer should be cleared after flush")

    @unittest.skipIf(
            True or # <<< HAS TO BE ENABLED BY USER TO AVOID CI LOGGING TO WANDB - also needs cuda
        not os.path.exists(os.path.expanduser("~/private/wandb_api.sh")),
        "WANDB API key file not found, skipping live logging test."
    )
    def test_live_logging(self):
        """Test actual logging to wandb if an API key file is found."""
        api_key_path = os.path.expanduser("~/private/wandb_api.sh")
        wandb_wrapper.init(project="test_project", wandb_api_key=None)

        # Log a test metric and flush
        for i in range(10):
            wandb_wrapper.log("test_metric", 123.456-i)
            # fill a torch tensor that resides on a different device
            # to test if the logging is device agnostic
            t = torch.rand(10, 10, device="cuda")
            t = torch.sum(t) #add some possibly lazy operation
            wandb_wrapper.log("torch_metric", t)

            wandb_wrapper.flush(prefix="test_")


        # Check that the log buffer is cleared after flushing
        self.assertEqual(len(wandb_wrapper.log_buffer), 0, "Log buffer should be cleared after flush")


if __name__ == "__main__":
    unittest.main()
