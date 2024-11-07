
from .wandb_tools import wandb_wrapper
import torch

class LoggingModule(torch.nn.Module):
    """
    torch.nn.Module class with integrated logging capabilities. Logs can be
    selectively enabled or disabled for this module and any nested
    submodules.

    Args:
        is_logging_module (bool): Set to True to enable logging for this module.

    Methods:
        _log(metric_name, value): Logs a metric if logging is enabled.
        _no_op(metric_name, value): No-op function for disabled logging.
        switch_logging(enable_logging): Enables or disables logging for
            this module and all nested LoggingModule instances.
    """

    _instance_count = 0 

    def __init__(self,  name=None, is_logging_module=False):
        super(LoggingModule, self).__init__()

        # Assign a unique name if none is provided
        if name is None:
            LoggingModule._instance_count += 1
            self.name = f"LoggingModule{LoggingModule._instance_count}"
        else:
            self.name = name

        self.switch_logging(is_logging_module)

    def _log(self, metric_name, value, skip_prefix=False):
        """
        Logs a metric using the wandb wrapper.

        Args:
            metric_name (str): The name of the metric.
            value (float): The value of the metric.
            skip_prefix (bool): If True, skips prefixing the metric name 
                                with the module's name. (Default: False)
        """
        # Prefix the metric name with the module's name
        if not skip_prefix:
            metric_name = f"{self.name}_{metric_name}"
        wandb_wrapper.log(metric_name, value)

    def _no_op(self, metric_name, value):
        """
        No-op function that does nothing, used when logging is disabled.
        """
        pass

    def switch_logging(self, enable_logging):
        """
        Enables or disables logging for this module and all nested submodules.

        Args:
            enable_logging (bool): True to enable logging, False to disable it.
        """
        self.is_logging_module = enable_logging
        self.log = self._log if enable_logging else self._no_op
        # Recursively apply to all child modules
        for child in self.children():
            if isinstance(child, LoggingModule):
                child.switch_logging(enable_logging)


    @staticmethod
    def switch_all_logging(module, enable_logging):
        """
        Searches through a given torch.nn.Module and applies switch_logging to any
        LoggingModule submodules found, enabling or disabling logging as specified.

        Args:
            module (torch.nn.Module): The module to search through.
            enable_logging (bool): True to enable logging, False to disable it.
        """
        for child in module.modules():
            if isinstance(child, LoggingModule):
                child.switch_logging(enable_logging)



class LossModule(LoggingModule):
    def __init__(self, name=None, is_logging_module=False, is_loss_active=True):
        """
        A PyTorch module designed to compute and record individual loss terms, inheriting from LoggingModule.
        This module allows toggling loss calculation on or off for efficient handling of multiple loss terms within
        complex model structures. Each LossModule instance stores its own computed losses, which can later be aggregated
        across a model.
    
        Attributes:
            _losses (list): An instance-level list that stores computed losses for the module, enabling
                            retrieval and aggregation of losses when needed.
            is_loss_active (bool): A property that returns whether loss calculation is enabled or disabled.
    
        Args:
            name (str, optional): Optional name for the module. If None, a unique name will be assigned.
            is_logging_module (bool): If True, enables logging for this module.
            is_loss_active (bool): If True, enables loss calculation for this module. Default is True.
    
        Methods:
            forward(*args, **kwargs): Computes the loss by calling compute_loss and appends it to the instance's loss list.
                                      This method is dynamically reassigned based on the `is_loss_active` state.
            compute_loss(*args, **kwargs): Placeholder for the actual loss computation. Should be implemented in subclasses.
                                           Returns a single scalar tensor representing the loss.
            switch_loss_calculation(enable_loss): Enables or disables loss calculation, dynamically assigning forward to
                                                  either compute_loss or a no-op function for JIT compatibility.
            clear_losses(): Clears all recorded losses for the instance, useful for resetting losses after aggregation.
            sum_all_losses(module): Recursively collects and sums losses from all LossModule instances within a given module.
                                    Returns the total loss as a single scalar tensor.
            switch_all_losses(module, enable_loss): Recursively enables or disables loss calculation for all LossModule
                                                    instances within a given module.
        """
        super(LossModule, self).__init__(name=name, is_logging_module=is_logging_module)
        self._losses = []  # Instance-level list to store losses for this LossModule
        self.switch_loss_calculation(is_loss_active)

    @property
    def is_loss_active(self):
        """Read-only property to access the loss calculation state."""
        return self.forward == self._loss_op

    def _loss_no_op(self, *args, **kwargs):
        """No-op function when loss calculation is disabled."""
        pass

    def _loss_op(self, *args, **kwargs):
        """Compute the loss and append to the instance's loss list."""
        loss = self.compute_loss(*args, **kwargs)
        self._losses.append(loss)

    def compute_loss(self, *args, **kwargs):
        """
        Placeholder for the actual loss computation. Should be implemented in subclasses.
        This function will be called by `forward` when the loss calculation is enabled.

        Must return a single scalar tensor representing the loss.

        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses of LossModule must implement the compute_loss method.")

    def switch_loss_calculation(self, enable_loss):
        """
        Enables or disables the loss calculation for this module, dynamically setting `forward` to either
        `_loss_op` (enabled) or `_loss_no_op` (disabled) for JIT compatibility.

        Args:
            enable_loss (bool): True to enable loss calculation, False to disable it.
        """
        self.forward = self._loss_op if enable_loss else self._loss_no_op

        # Recursively apply to all child modules
        for child in self.children():
            if isinstance(child, LossModule):
                child.switch_loss_calculation(enable_loss)


    @staticmethod
    def switch_all_losses(module, enable_loss):
        """
        Searches through a given torch.nn.Module and applies switch_loss_calculation to any
        LossModule submodules found, enabling or disabling loss calculation as specified.

        Args:
            module (torch.nn.Module): The module to search through.
            enable_loss (bool): True to enable loss calculation, False to disable it.
        """
        for child in module.modules():
            if isinstance(child, LossModule):
                child.switch_loss_calculation(enable_loss)

    def clear_losses(self):
        """Clears the accumulated losses in this module's instance-level loss list."""
        self._losses.clear()

    @staticmethod
    def sum_all_losses(module):
        """
        Recursively collects and sums all losses from LossModule instances within a given module.

        Args:
            module (torch.nn.Module): The module to search through.

        Returns:
            torch.Tensor: The sum of all accumulated losses from LossModule instances.
        """
        total_loss = torch.tensor(0.0, requires_grad=True)
        for child in module.modules():
            if isinstance(child, LossModule):
                if child._losses:
                    total_loss = total_loss + sum(child._losses)
        return total_loss
    
    @staticmethod
    def clear_all_losses(module):
        """
        Recursively clears all accumulated losses from LossModule instances within a given module.

        Args:
            module (torch.nn.Module): The module to search through.
        """
        for child in module.modules():
            if isinstance(child, LossModule):
                child.clear_losses()
