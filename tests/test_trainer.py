import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from djctools.training import Trainer
from djctools.module_extensions import LossModule
from djctools.wandb_tools import wandb_wrapper

try:
    import djcdata
    djcdata_available = True
except ImportError:
    djcdata_available = False



class SimpleLossModule(LossModule):
    """A simple loss module for testing purposes, inheriting LossModule."""
    def compute_loss(self, predictions, targets):
        """Compute a simple MSE loss for testing."""
        return nn.functional.mse_loss(predictions, targets)
    
class SimpleModel(torch.nn.Module):
    """A simple model for testing purposes, inheriting LossModule."""
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 1)
        )
        
        self.loss = SimpleLossModule(logging_active=True, name="SimpleLossModule")
        self.triple_input = False

    def forward(self, data):
        if not self.triple_input:
            x, target = data['inputs'], data['targets']
        else:
            # this is following the TrainData_mock definition in djcdata
            x, target = data[0]["features_ragged"], data[1]["truth_ragged"]
        x = self.fc(x)
        self.loss(x, target)
        return x
        


class DummyDataLoader:
    """A dummy data loader that yields random data for testing purposes."""
    def __init__(self, num_batches, batch_size):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.current_batch = 0

    def __iter__(self):
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
        self.current_batch += 1
        inputs = torch.randn(self.batch_size, 10)
        targets = torch.randn(self.batch_size, 1)*0.1 + inputs.sum(dim=1, keepdim=True)
        return {'inputs': inputs, 'targets': targets}


class TestTrainer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup required to ensure only one instance runs at a time
        print("Setting up sequential class resources")

    @classmethod
    def tearDownClass(cls):
        # Teardown actions
        print("Tearing down sequential class resources")

    def _setUp(self, num_gpus):
        """Set up the model, optimizer, and trainer for each test."""
        self.model = SimpleModel()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.trainer = Trainer(model=self.model, optimizer=self.optimizer, num_gpus=num_gpus, verbose_level=1)  # Use CPU for testing
        self.train_loader = DummyDataLoader(num_batches=500, batch_size=1024)
        self.val_loader = DummyDataLoader(num_batches=300, batch_size=1024)
        wandb_wrapper.activate()  # Activate wandb for testing, don't initialise the connection though

    def test_initialization(self):
        self._setUp(num_gpus=0)
        """Test if Trainer initializes correctly with the model and optimizer."""
        self.assertIsInstance(self.trainer.model, SimpleModel)
        self.assertIsInstance(self.trainer.optimizer, optim.SGD)
        self.assertEqual(self.trainer.num_gpus, 0)  # Should default to CPU for testing

    def test_single_cpu_training(self):
        self._setUp(num_gpus=0)
        """Test training loop on a single GPU or CPU."""
        self.trainer.train_loop(self.train_loader)

    def test_single_gpu_training(self):
        self._setUp(num_gpus=1)
        """Test training loop on a single GPU or CPU."""
        self.trainer.train_loop(self.train_loader)
        self.trainer.train_loop(self.train_loader)

    def test_multi_gpu_training(self):
        self._setUp(num_gpus=3)
        """Test training loop on a single GPU or CPU."""
        self.trainer.train_loop(self.train_loader)
        self.trainer.train_loop(self.train_loader)

    def test_validation_loop(self):
        self._setUp(num_gpus=0)
        """Test validation loop execution and logging of validation losses."""
        self.trainer.val_loop(self.val_loader)
        self.trainer.val_loop(self.val_loader)

    def test_multi_gpu_validation(self):
        self._setUp(num_gpus=3)
        """Test validation loop execution and logging of validation losses."""
        self.trainer.val_loop(self.val_loader)
        self.trainer.val_loop(self.val_loader)

    def test_save_and_load_model(self):
        self._setUp(num_gpus=0)
        """Test saving and loading of model weights."""
        filepath = "test_model.pth"
        self.trainer.save_model(filepath)
        
        # Create a new instance and load weights
        model2 = SimpleModel()
        optimizer2 = optim.SGD(model2.parameters(), lr=0.01)
        trainer2 = Trainer(model=model2, optimizer=optimizer2, num_gpus=0, verbose_level=1)
        trainer2.load_model(filepath)

        # Check if weights are loaded correctly
        for param1, param2 in zip(self.model.parameters(), model2.parameters()):
            self.assertTrue(torch.equal(param1, param2))


    def do_test_with_djcdata(self, num_gpus):
        self._setUp(num_gpus=num_gpus)
        #overwrite the data loaders
        from djcdata import TrainDataGenerator
        TrainDataGenerator.debuglevel = 3
        from djcdata.torch_interface import MockDJCDataLoader
        train_loader = MockDJCDataLoader(batch_size=1500, dict_output=True)
        self.model.triple_input = True #make the model compatible in a simple way

        self.trainer.train_loop(train_loader)

    
    @unittest.skipIf(not djcdata_available, "djcdata not available")
    def test_with_djcdata_single_cpu(self):
        self.do_test_with_djcdata(0)

    @unittest.skipIf(not djcdata_available, "djcdata not available")
    def test_with_djcdata_single_gpu(self):
        self.do_test_with_djcdata(1)

    @unittest.skipIf(not djcdata_available, "djcdata not available")
    def test_with_djcdata_multi_gpu(self):
        self.do_test_with_djcdata(3)

    def tearDown(self):
        """Clean up any files created during testing."""
        import os
        if os.path.exists("test_model.pth"):
            os.remove("test_model.pth")


if __name__ == "__main__":
    unittest.main()
