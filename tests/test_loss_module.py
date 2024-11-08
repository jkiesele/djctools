import unittest
import torch
from djctools.module_extensions import LossModule, sum_all_losses, clear_all_losses, switch_all_losses, switch_all_logging

# Define a simple custom loss class for testing purposes
class TestLossModule(LossModule):
    def compute_loss(self, predictions, targets):
        # Use Mean Squared Error for simplicity
        loss = torch.nn.functional.mse_loss(predictions, targets)
        self.log('loss', loss)
        return loss

class LossModuleTest(unittest.TestCase):
    
    def setUp(self):
        """Set up a model with two loss modules for testing."""
        class TestModel(torch.nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.loss1 = TestLossModule(logging_active=False, loss_active=True, name="loss1")
                self.loss2 = TestLossModule(logging_active=False, loss_active=True, name="loss2")

            def forward(self, predictions, targets=None):
                self.loss1(predictions, targets)
                self.loss2(predictions, targets)
                return predictions

        self.model = TestModel()
    
    def test_loss_computation_and_storage(self):
        """Test that losses are computed and stored in the loss list."""
        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)

        # Compute losses
        self.model(predictions, targets)

        # Check that losses were stored
        self.assertEqual(len(self.model.loss1._losses), 1)
        self.assertEqual(len(self.model.loss2._losses), 1)

    def test_switch_loss_calculation(self):
        """Test enabling and disabling loss calculation dynamically."""
        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)

        # Disable loss calculation for loss1
        self.model.loss1.switch_loss_calculation(False)
        self.assertFalse(self.model.loss1.loss_active)

        # Compute losses
        self.model(predictions, targets)

        # Ensure loss1 did not record a loss and loss2 did
        self.assertEqual(len(self.model.loss1._losses), 0)
        self.assertEqual(len(self.model.loss2._losses), 1)

        # Re-enable loss calculation and verify
        self.model.loss1.switch_loss_calculation(True)
        self.assertTrue(self.model.loss1.loss_active)
        self.model(predictions, targets)
        self.assertEqual(len(self.model.loss1._losses), 1)

        # Disable loss calculation for all losses
        switch_all_losses(self.model, False)
        self.assertFalse(self.model.loss1.loss_active)
        self.assertFalse(self.model.loss2.loss_active)

        # now the model should also work without giving any targets to it
        try:
            self.model(predictions)
        except:
            self.fail("Model should work without targets")

    def test_sum_all_losses(self):
        """Test that all accumulated losses are correctly summed."""
        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)

        # Compute losses for both modules
        self.model.loss1(predictions, targets)
        self.model.loss2(predictions, targets)

        # Sum all losses
        total_loss = sum_all_losses(self.model)
        clear_all_losses(self.model)

        # Check that total_loss is a single scalar tensor
        self.assertTrue(isinstance(total_loss, torch.Tensor))
        self.assertEqual(total_loss.shape, torch.Size([]))
        self.assertGreater(total_loss.item(), 0)

    def test_clear_all_losses(self):
        """Test that all accumulated losses are cleared correctly."""
        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)

        # Compute losses for both modules
        self.model(predictions, targets)

        # Ensure there are losses
        self.assertEqual(len(self.model.loss1._losses), 1)
        self.assertEqual(len(self.model.loss2._losses), 1)

        # Clear all losses
        clear_all_losses(self.model)

        # Verify losses are cleared
        self.assertEqual(len(self.model.loss1._losses), 0)
        self.assertEqual(len(self.model.loss2._losses), 0)

    def test_loss_active_property(self):
        """Test that the loss_active property correctly reflects the module's active state."""
        # Initially active
        self.assertTrue(self.model.loss1.loss_active)
        self.assertTrue(self.model.loss2.loss_active)

        # Disable loss calculation for loss1 and verify
        self.model.loss1.switch_loss_calculation(False)
        self.assertFalse(self.model.loss1.loss_active)
        self.assertTrue(self.model.loss2.loss_active)

        # Re-enable and verify
        self.model.loss1.switch_loss_calculation(True)
        self.assertTrue(self.model.loss1.loss_active)

    def test_switch_all_losses(self):
        """Test that all losses can be enabled or disabled at once."""
        # Initially active
        self.assertTrue(self.model.loss1.loss_active)
        self.assertTrue(self.model.loss2.loss_active)

        # Disable all losses
        switch_all_losses(self.model, False)
        self.assertFalse(self.model.loss1.loss_active)
        self.assertFalse(self.model.loss2.loss_active)

        # Re-enable all losses
        switch_all_losses(self.model, True)
        self.assertTrue(self.model.loss1.loss_active)
        self.assertTrue(self.model.loss2.loss_active)

    def test_switch_logging_on_loss_module(self):
        """Test that logging can be enabled or disabled for a loss module."""
        # Initially disabled
        self.assertFalse(self.model.loss1.logging_active)

        # Enable logging
        self.model.loss1.switch_logging(True)
        self.assertTrue(self.model.loss1.logging_active)
        # make sure loss is still active
        self.assertTrue(self.model.loss1.loss_active)

        # Disable logging
        self.model.loss1.switch_logging(False)
        self.assertFalse(self.model.loss1.logging_active)
        # make sure loss is still active
        self.assertTrue(self.model.loss1.loss_active)

    def test_switch_logging_on_all_losses(self):
        """Test that logging can be enabled or disabled for all loss modules at once."""
        # Initially disabled
        self.assertFalse(self.model.loss1.logging_active)
        self.assertFalse(self.model.loss2.logging_active)

        # Enable logging for all losses
        switch_all_logging(self.model, True)
        self.assertTrue(self.model.loss1.logging_active)
        self.assertTrue(self.model.loss2.logging_active)

        # Disable logging for all losses
        switch_all_logging(self.model, False)
        self.assertFalse(self.model.loss1.logging_active)
        self.assertFalse(self.model.loss2.logging_active)
        # make sure loss is still active
        self.assertTrue(self.model.loss1.loss_active)


if __name__ == "__main__":
    unittest.main()
