from .wandb_tools import wandb_wrapper
import torch

class LoggingModule(torch.nn.Module):
    """
    LoggingModule integrates logging capabilities into PyTorch modules, allowing 
    selective logging for metrics in nested module structures.

    It is a torch.nn.Module class with integrated logging capabilities. Logs can be
    selectively enabled or disabled for this module and any nested LoggingModule
    instances.

    Args:
        logging_active (bool): Set to True to enable logging for this module.

    User methods:
        log(metric_name, value): Logs a metric if logging is enabled, otherwise does nothing.
                                 Value can be a float, int, or tensor on any device. No need to break
                                 jit-compatibility by converting to numpy or calling .item() here.
        compute_metrics(*args, **kwargs): Should be implemented in subclasses.
                                          If logging is enabled, this function will be called by the forward method.
                                          If logging is disabled, this function will not be called at all.
        switch_logging(enable_logging): Enables or disables logging for
                                        this module and all nested LoggingModule instances.
    """

    _instance_count = 0 

    def __init__(self, name=None, logging_active=False):
        super(LoggingModule, self).__init__()

        # Assign a unique name if none is provided
        if name is None:
            LoggingModule._instance_count += 1
            self.name = f"LoggingModule{LoggingModule._instance_count}"
        else:
            self.name = name

        self.switch_logging(logging_active)

    def _log(self, metric_name, value, skip_prefix=False):
        """
        Logs a metric using the wandb wrapper.

        Args:
            metric_name (str): The name of the metric.
            value (float): The value of the metric.
            skip_prefix (bool): If True, skips prefixing the metric name 
                                with the module's name (Default: False).
        
        Note:
            Prefixing the metric name with the module's name is helpful for 
            distinguishing metrics from different instances in nested structures.
        """
        if not skip_prefix:
            metric_name = f"{self.name}_{metric_name}"
        wandb_wrapper.log(metric_name, value)

    def _no_op(self, *args, **kwargs):
        """No-op function that does nothing, used when logging is disabled."""
        pass

    def compute_metrics(self, *args, **kwargs):
        """
        Placeholder for the actual metric computation.
        Should be implemented in subclasses.
        To log a metric, use the self.log function within this function.

        When logging is enabled, this function will replace the forward method.
        If logging is disabled, this function will not be called at all.
        
        This enables use of truth information when logging without requiring truth
        data in inference-only scenarios.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        
        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError("Subclasses of LoggingModule must implement the compute_metrics method.")

    def switch_logging(self, logging_active):
        """
        Enables or disables logging for this module and all nested LoggingModule instances.

        Args:
            logging_active (bool): True to enable logging, False to disable it.

        Note:
            When logging is enabled, `log` is set to `_log` and `forward` to `compute_metrics`.
            When disabled, both are set to `_no_op`. This approach supports JIT compatibility
            and prevents unnecessary computation.
        """
        self.log = self._log if logging_active else self._no_op
        self.forward = self.compute_metrics if logging_active else self._no_op
        for child in self.children():
            if isinstance(child, LoggingModule):
                child.switch_logging(logging_active)

    @property
    def logging_active(self):
        """Read-only property to access the logging state."""
        return self.log == self._log



class LossModule(LoggingModule):
    def __init__(self, name=None, logging_active=False, loss_active=True):
        """
        LossModule extends LoggingModule to enable modular loss computation, allowing 
        fine-grained control over loss terms in complex models.

        It is a PyTorch module designed to compute and record individual loss terms, inheriting from LoggingModule.
        This module allows toggling loss calculation on or off for efficient handling of multiple loss terms within
        complex model structures. Each LossModule instance stores its own computed losses, which can later be aggregated
        across a model.
    
        Attributes:
            _losses (list): An instance-level list that stores computed losses for the module, enabling
                            retrieval and aggregation of losses when needed.
            loss_active (bool): A property that returns whether loss calculation is enabled or disabled.
    
        Args:
            name (str, optional): Optional name for the module. If None, a unique name will be assigned.
            logging_active (bool): If True, enables logging for this module.
            loss_active (bool): If True, enables loss calculation for this module. Default is True.
    
        Methods:
            forward(*args, **kwargs): Computes the loss by calling compute_loss and appends it to the instance's loss list.
                                      This method is dynamically reassigned based on the `loss_active` state.
            compute_loss(*args, **kwargs): Should be implemented in subclasses.
                                           Returns a single scalar tensor representing the loss.
            switch_loss_calculation(enable_loss): Enables or disables loss calculation, dynamically assigning forward to
                                                  either compute_loss or a no-op function for JIT compatibility.
            clear_losses(): Clears all recorded losses for the instance, useful for resetting losses after aggregation.
            
            sum_all_losses(module): Recursively collects and sums losses from all LossModule instances within a given module.
                                    Returns the total loss as a single scalar tensor.
            switch_all_losses(module, enable_loss): Recursively enables or disables loss calculation for all LossModule
                                                    instances within a given module.
        """
        super(LossModule, self).__init__(name=name, logging_active=logging_active)
        self._losses = []  # Instance-level list to store losses for this LossModule
        self.switch_loss_calculation(loss_active)

    @property
    def loss_active(self):
        """Read-only property to access the loss calculation state."""
        return self.forward == self._compute_loss_and_record

    def _compute_loss_and_record(self, *args, **kwargs):
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
    
    def switch_logging(self, logging_active):
        """
        Enables or disables logging for this module and all nested submodules.
        This only affects calls to the `log` method, not the forward method.

        Args:
            logging_active (bool): True to enable logging, False to disable it.
        """
        self.log = self._log if logging_active else self._no_op
        for child in self.children():
            if isinstance(child, LoggingModule): # now these are all nested logging modules
                child.switch_logging(logging_active)

    def switch_loss_calculation(self, loss_active):
        """
        Enables or disables the loss calculation for this module, dynamically setting `forward` to either
        `_compute_loss_and_record` (enabled) or `_no_op` (disabled) for JIT compatibility.

        Args:
            loss_active (bool): True to enable loss calculation, False to disable it.

        Note:
            This method applies recursively to all nested LossModule instances within the module.
        """
        self.forward = self._compute_loss_and_record if loss_active else self._no_op

        # Recursively apply to all child modules
        for child in self.children():
            if isinstance(child, LossModule):
                child.switch_loss_calculation(loss_active)

    def clear_losses(self):
        """Clears the accumulated losses in this module's instance-level loss list."""
        self._losses.clear()


## functions for model-wide application

def switch_all_logging(module : torch.nn.Module, logging_active : bool):
    """
    Searches through a given torch.nn.Module and applies switch_logging to any
    LoggingModule submodules found, enabling or disabling logging as specified.
    This is done recursively across all levels of nested LoggingModule instances.

    Args:
        module (torch.nn.Module): The module to search through.
        logging_active (bool): True to enable logging, False to disable it.
    """
    for child in module.modules():
        if isinstance(child, LoggingModule):
            child.switch_logging(logging_active)

def switch_all_losses(module : torch.nn.Module, loss_active : bool):
    """
    Searches through a given torch.nn.Module and applies switch_loss_calculation to any
    LossModule submodules found, enabling or disabling loss calculation as specified.
    This is done recursively across all levels of nested LossModule instances.

    Args:
        module (torch.nn.Module): The module to search through.
        loss_active (bool): True to enable loss calculation, False to disable it.
    """
    for child in module.modules():
        if isinstance(child, LossModule):
            child.switch_loss_calculation(loss_active)

def sum_all_losses(module : torch.nn.Module):
    """
    Recursively collects and sums all losses from LossModule instances within a given module.

    Args:
        module (torch.nn.Module): The module to search through.

    Returns:
        torch.Tensor: A single scalar tensor representing the sum of all accumulated losses.
    
    Note:
        This method operates recursively across all levels of nested LossModule instances.
    """
    if hasattr(module, 'parameters') and next(module.parameters(), None) is not None:
        device = next(module.parameters()).device
    else:
        device = torch.device('cpu')
    total_loss = torch.tensor(0.0, requires_grad=True).to(device)
    
    for child in module.modules():
        if isinstance(child, LossModule):
            if child._losses:
                total_loss = total_loss + sum([l.to(device) for l in child._losses])
    return total_loss

def clear_all_losses(module : torch.nn.Module):
    """
    Recursively clears all accumulated losses from LossModule instances within a given module.

    Args:
        module (torch.nn.Module): The module to search through.
    """
    for child in module.modules():
        if isinstance(child, LossModule):
            child.clear_losses()