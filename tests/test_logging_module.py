import unittest
from djctools.module_extensions import LoggingModule
from djctools.wandb_tools import wandb_wrapper


class TestLoggingModule(unittest.TestCase):

    def setUp(self):
        """Reset the wandb wrapper before each test and clear the instance count."""
        wandb_wrapper.log_buffer.clear()
        wandb_wrapper.active = True
        wandb_wrapper.initialized = False
        LoggingModule._instance_count = 0  # Reset instance count to ensure unique names

    def test_default_name_assignment(self):
        """Test that modules are assigned unique default names when name is not provided."""
        module1 = LoggingModule()
        module2 = LoggingModule()
        self.assertEqual(module1.name, "LoggingModule1")
        self.assertEqual(module2.name, "LoggingModule2")

    def test_custom_name_assignment(self):
        """Test that custom names are assigned correctly."""
        module = LoggingModule(name="CustomModule")
        self.assertEqual(module.name, "CustomModule")

    def test_logging_enabled(self):
        """Test that metrics are logged when logging is enabled."""
        module = LoggingModule(logging_active=True)
        module.log("test_metric", 42)
        self.assertIn("LoggingModule1_test_metric", wandb_wrapper.log_buffer)
        self.assertEqual(wandb_wrapper.log_buffer["LoggingModule1_test_metric"], 42)

    def test_logging_disabled(self):
        """Test that metrics are not logged when logging is disabled."""
        module = LoggingModule(logging_active=False)
        module.log("test_metric", 42)
        self.assertNotIn("LoggingModule1_test_metric", wandb_wrapper.log_buffer)

    def test_switch_logging(self):
        """Test that switch_logging properly toggles logging on and off."""
        module = LoggingModule(logging_active=False)
        module.switch_logging(True)
        module.log("test_metric_on", 1)
        self.assertIn("LoggingModule1_test_metric_on", wandb_wrapper.log_buffer)

        # Now switch off logging and check that new logs are not recorded
        module.switch_logging(False)
        module.log("test_metric_off", 2)
        self.assertNotIn("LoggingModule1_test_metric_off", wandb_wrapper.log_buffer)

    def test_nested_module_logging(self):
        """Test that nested modules respect the logging state of the parent."""
        parent_module = LoggingModule(logging_active=True)
        child_module = LoggingModule(logging_active=True)
        parent_module.add_module("child", child_module)

        # Log a metric in the parent and child modules
        parent_module.log("parent_metric", 100)
        child_module.log("child_metric", 200)

        # Check both metrics in the log buffer
        self.assertIn("LoggingModule1_parent_metric", wandb_wrapper.log_buffer)
        self.assertEqual(wandb_wrapper.log_buffer["LoggingModule1_parent_metric"], 100)
        self.assertIn("LoggingModule2_child_metric", wandb_wrapper.log_buffer)
        self.assertEqual(wandb_wrapper.log_buffer["LoggingModule2_child_metric"], 200)

    def test_switch_logging_in_nested_module(self):
        """Test that switch_logging propagates to nested modules."""
        parent_module = LoggingModule(logging_active=True)
        child_module = LoggingModule(logging_active=True)
        parent_module.add_module("child", child_module)

        # Disable logging in the parent; child should also stop logging
        parent_module.switch_logging(False)
        parent_module.log("parent_metric", 123)
        child_module.log("child_metric", 456)

        # Neither metric should be in the log buffer
        self.assertNotIn("LoggingModule1_parent_metric", wandb_wrapper.log_buffer)
        self.assertNotIn("LoggingModule2_child_metric", wandb_wrapper.log_buffer)

        # Re-enable logging; both parent and child should log again
        parent_module.switch_logging(True)
        parent_module.log("parent_metric_enabled", 789)
        child_module.log("child_metric_enabled", 1011)

        self.assertIn("LoggingModule1_parent_metric_enabled", wandb_wrapper.log_buffer)
        self.assertIn("LoggingModule2_child_metric_enabled", wandb_wrapper.log_buffer)

if __name__ == "__main__":
    unittest.main()
