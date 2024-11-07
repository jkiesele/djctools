# mnist_trainer_example.py

#try to import torchvision and print warning if does not exist
try:
    import torchvision
except ImportError:
    print("Warning: torchvision not found. Please install torchvision to run this example.")
    exit()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Import the djctools components
from djctools.module_extensions import LossModule
from djctools.module_extensions import LoggingModule
from djctools.training import Trainer
from djctools.wandb_tools import wandb_wrapper


# Define a custom LossModule
class MNISTLossModule(LossModule):
    def __init__(self, **kwargs):
        super(MNISTLossModule, self).__init__(**kwargs)
        self.criterion = nn.CrossEntropyLoss()
    
    def compute_loss(self, outputs, targets):
        loss = self.criterion(outputs, targets)
        # Log the loss value, the name of the module 
        # will be prepended to the metric name
        self.log("loss", loss.item())
        return loss

# Define the model
class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.loss_module = MNISTLossModule(is_logging_module=True, name="MNISTLossModule")
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 14 * 14, 10)
    
    def forward(self, data):
        # for pure inference, targets can be None and 
        # the loss will not be computed if the loss modules 
        # are all turned off
        inputs, targets = data
        x = inputs
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 14 * 14)
        outputs = self.fc1(x)
        self.loss_module(outputs, targets)
        return outputs

def main():

    # Initialize wandb through the wandb_wrapper
    wandb_wrapper.init(project="mnist_trainer_example")

    # Set device to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    # each GPU will receive a batch of 64 samples
    batch_size = 128 
    num_epochs = 2
    learning_rate = 0.001
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_gpus = min(num_gpus, 3)  # Limit to 3 GPUs for testing
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize the model and optimizer
    model = MNISTModel()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize the Trainer, the total loss is always logged if wandb is enabled
    trainer = Trainer(model, optimizer, 
                      num_gpus=num_gpus)
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch+1}")
        trainer.train_loop(train_loader)
        trainer.val_loop(val_loader)
    
    # turn off the losses for the model, this also applies to nested modules
    LossModule.switch_all_losses(model, False)
    # turn off all logging, this also applies to nested modules
    LoggingModule.switch_all_logging(model, False)

    # Save the trained model for inference
    trainer.save_model("mnist_model.pth")
    
    # Finish the wandb run
    wandb_wrapper.finish()

if __name__ == "__main__":
    main()
