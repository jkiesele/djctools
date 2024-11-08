import wandb
import torch
import os

class _WandbWrapper:
    """
    Singleton wrapper around wandb that buffers log entries and controls logging activation for a model.
    This wrapper enables logging only when activated and initializes wandb on demand. It attempts to load
    the wandb API key from '~/private/wandb_api.sh' if an API key is not directly provided.

    Attributes:
        log_buffer (dict): Temporary storage for log entries, cleared after each flush.
        active (bool): Controls whether logging is active. When False, all logging calls are ignored.
        initialized (bool): Indicates whether wandb has been initialized to prevent multiple initializations.

    Methods:
        activate(): Enables logging, allowing metrics to be recorded.
        deactivate(): Disables logging, preventing metrics from being recorded.
        init(*args, wandb_api_key=None, **kwargs): Initializes wandb, attempting to load an API key from
                                                   '~/private/wandb_api.sh' if not provided.
        log(metric_name, value): Buffers a metric for logging. Metrics are stored in `log_buffer` until flushed.
        flush(prefix=""): Flushes the buffered logs to wandb. Each metric name can be prefixed for easy identification.
        finish(): Ends the wandb run, finalizing any remaining logging activities.

    API Key Loading:
        If the `wandb_api_key` parameter is not provided when calling `init`, this class will attempt to find
        and load the API key from a file located at '~/private/wandb_api.sh'. The file should contain a line in the
        following format:
        
            WANDB_API_KEY="your_api_key_here"

        The class will read this file, extract the key, and set it in the environment variable `WANDB_API_KEY`.
        This environment variable is recognized by wandb, allowing it to initialize automatically without requiring
        the API key to be passed manually each time.

        If the file is not found or does not contain the key, a warning message will be displayed. You can also
        provide the API key directly by passing it to the `wandb_api_key` parameter in the `init()` method, which will
        take precedence over the file.

    Usage Example:

        # Optionally activate logging
        wandb_wrapper.activate()

        # Initialize wandb, attempting to load API key from file if not provided
        wandb_wrapper.init(project="example_project", wandb_api_key="optional_key")

        # Log metrics
        wandb_wrapper.log("accuracy", 0.95)
        wandb_wrapper.log("loss", 0.05)

        # Flush buffered logs to wandb
        wandb_wrapper.flush()
        
        # Finish the wandb session
        wandb_wrapper.finish()
    """

    _instance = None  # Class-level attribute to hold the single instance

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(_WandbWrapper, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.log_buffer = {}
            self.active = True
            self.initialized = False

    def activate(self):
        """Enable logging, allowing metrics to be recorded."""
        self.active = True

    def deactivate(self):
        """Disable logging, preventing metrics from being recorded."""
        self.active = False

    def init(self, *args, wandb_api_key=None, **kwargs):
        """
        Explicitly initialize wandb, separating it from object creation.

        Args:
            *args: Positional arguments for wandb initialization.
            wandb_api_key (str, optional): API key for wandb. If None, attempts to load from '~/private/wandb_api.sh'.
            **kwargs: Additional keyword arguments for wandb initialization.

        API Key Loading:
            If `wandb_api_key` is not provided, this method will attempt to load it from a file located at
            '~/private/wandb_api.sh'. The file should contain a line formatted as:

                WANDB_API_KEY="your_api_key_here"

            If the API key is found in this file, it will be set in the environment variable `WANDB_API_KEY` for
            wandb to use automatically. If the file is missing or does not contain the key, a warning is displayed.

        Example:
            wandb_wrapper.init(project="my_project")
        """
        if self.active and not self.initialized:
            if wandb_api_key is None:
                # Attempt to load the API key from '~/private/wandb_api.sh'
                api_key_path = os.path.expanduser('~/private/wandb_api.sh')
                if os.path.exists(api_key_path):
                    with open(api_key_path, 'r') as f:
                        for line in f:
                            if "WANDB_API_KEY=" in line:
                                # Extract the key and set it in the environment
                                wandb_api_key = line.strip().split('=')[1].strip('"').strip("'")
                                os.environ["WANDB_API_KEY"] = wandb_api_key
                                break
                else:
                    print("Warning: API key file '~/private/wandb_api.sh' not found.")

            if wandb_api_key:
                os.environ["WANDB_API_KEY"] = wandb_api_key

            # Initialize wandb with the provided arguments
            wandb.init(*args, **kwargs)
            self.initialized = True

    def log(self, metric_name, value):
        """
        Buffer a metric for logging. Metrics are stored in `log_buffer` until flushed.
        This maintains jit compatibility by avoiding direct logging calls or tensor conversions.

        Args:
            metric_name (str): The name of the metric.
            value (float, int, or Tensor): The value of the metric. Tensors will be converted to floats/ints in `flush()`.
        """
        if self.active:
            self.log_buffer[metric_name] = value

    def flush(self, prefix=""):
        """
        Flush the buffered logs to wandb. Each metric name is prefixed with `prefix` to avoid conflicts.

        Args:
            prefix (str): A string prefix for each metric name.
        """
        if self.active and self.initialized:
            for metric_name, value in self.log_buffer.items():
                # Convert tensors to floats if needed before logging
                if isinstance(value, torch.Tensor):
                    value = value.item()
                wandb.log({prefix + metric_name: value})
        self.log_buffer.clear()

    def finish(self):
        """
        Finish the wandb run, closing the active run if one exists.
        """
        if self.active and self.initialized:
            wandb.finish()

# Instantiate a single instance for the package
wandb_wrapper = _WandbWrapper()
