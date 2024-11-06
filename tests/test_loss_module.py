import unittest
import torch
from djctools.module_extensions import LossModule

# Define a simple custom loss class for testing purposes
class TestLossModule(LossModule):
    def compute_loss(self, predictions, targets):
        # Use Mean Squared Error for simplicity
        loss = torch.nn.functional.mse_loss(predictions, targets)
        return loss

class LossModuleTest(unittest.TestCase):
    
    def setUp(self):
        """Set up a model with two loss modules for testing."""
        self.model = torch.nn.Module()
        self.model.loss1 = TestLossModule(is_logging_module=False, is_loss_active=True)
        self.model.loss2 = TestLossModule(is_logging_module=False, is_loss_active=True)
    
    def test_loss_computation_and_storage(self):
        """Test that losses are computed and stored in the loss list."""
        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)

        # Compute losses
        self.model.loss1(predictions, targets)
        self.model.loss2(predictions, targets)

        # Check that losses were stored
        self.assertEqual(len(self.model.loss1._losses), 1)
        self.assertEqual(len(self.model.loss2._losses), 1)

    def test_switch_loss_calculation(self):
        """Test enabling and disabling loss calculation dynamically."""
        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)

        # Disable loss calculation for loss1
        self.model.loss1.switch_loss_calculation(False)
        self.assertFalse(self.model.loss1.is_loss_active)

        # Compute losses
        self.model.loss1(predictions, targets)
        self.model.loss2(predictions, targets)

        # Ensure loss1 did not record a loss and loss2 did
        self.assertEqual(len(self.model.loss1._losses), 0)
        self.assertEqual(len(self.model.loss2._losses), 1)

        # Re-enable loss calculation and verify
        self.model.loss1.switch_loss_calculation(True)
        self.assertTrue(self.model.loss1.is_loss_active)
        self.model.loss1(predictions, targets)
        self.assertEqual(len(self.model.loss1._losses), 1)

    def test_sum_all_losses(self):
        """Test that all accumulated losses are correctly summed."""
        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)

        # Compute losses for both modules
        self.model.loss1(predictions, targets)
        self.model.loss2(predictions, targets)

        # Sum all losses
        total_loss = LossModule.sum_all_losses(self.model)

        # Check that total_loss is a single scalar tensor
        self.assertTrue(isinstance(total_loss, torch.Tensor))
        self.assertEqual(total_loss.shape, torch.Size([]))
        self.assertGreater(total_loss.item(), 0)

    def test_clear_all_losses(self):
        """Test that all accumulated losses are cleared correctly."""
        predictions = torch.randn(10, 5)
        targets = torch.randn(10, 5)

        # Compute losses for both modules
        self.model.loss1(predictions, targets)
        self.model.loss2(predictions, targets)

        # Ensure there are losses
        self.assertEqual(len(self.model.loss1._losses), 1)
        self.assertEqual(len(self.model.loss2._losses), 1)

        # Clear all losses
        LossModule.clear_all_losses(self.model)

        # Verify losses are cleared
        self.assertEqual(len(self.model.loss1._losses), 0)
        self.assertEqual(len(self.model.loss2._losses), 0)

    def test_is_loss_active_property(self):
        """Test that the is_loss_active property correctly reflects the module's active state."""
        # Initially active
        self.assertTrue(self.model.loss1.is_loss_active)
        self.assertTrue(self.model.loss2.is_loss_active)

        # Disable loss calculation for loss1 and verify
        self.model.loss1.switch_loss_calculation(False)
        self.assertFalse(self.model.loss1.is_loss_active)
        self.assertTrue(self.model.loss2.is_loss_active)

        # Re-enable and verify
        self.model.loss1.switch_loss_calculation(True)
        self.assertTrue(self.model.loss1.is_loss_active)


if __name__ == "__main__":
    unittest.main()
