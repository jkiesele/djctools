import unittest
import torch
from djctools.module_extensions import PlottingModule, switch_all_plotting
import time

# Mock subclass for testing
class MockPlottingModule(PlottingModule):
    def __init__(self, *args, **kwargs):
        super(MockPlottingModule, self).__init__(*args, **kwargs)
        self.plotted_data = []  # Store data passed to plot for validation

    def plot(self, data):
        self.plotted_data.append(data)  # Mock plotting by storing the data
        time.sleep(0.5)  # Simulate a slow plotting operation


# Unit tests
class TestPlottingModule(unittest.TestCase):
    def test_initial_state(self):
        module = MockPlottingModule(plotting_active=True)
        self.assertTrue(module.plotting_active)
        self.assertEqual(module._cache, [])
        self.assertIsNone(module._plot_thread)

    def test_forward_caches_data(self):
        module = MockPlottingModule(plotting_active=True)
        module.forward(1, 2, a=3)
        self.assertEqual(len(module._cache), 1)
        self.assertEqual(module._cache[0], ((1, 2), {'a': 3}))

    def test_forward_ignores_data_when_disabled(self):
        module = MockPlottingModule(plotting_active=False)
        module.forward(1, 2, a=3)
        self.assertEqual(len(module._cache), 0)

    def test_flush_starts_thread(self):
        module = MockPlottingModule(plotting_active=True)
        module.forward(1, 2)
        module.flush()
        module._join_plot_thread() #make sure thread is joined in this test
        self.assertEqual(len(module.plotted_data), 1)
        self.assertEqual(module.plotted_data[0], [((1, 2), {})])

    def test_flush_skips_if_no_data(self):
        module = MockPlottingModule(plotting_active=True)
        with self.assertLogs(level="WARNING") as log:
            module.flush()
        self.assertIn("flush called with no data cached despite plotting being active", log.output[0])

    def test_flush_skips_if_thread_running(self):
        module = MockPlottingModule(plotting_active=True)
        module.forward(1, 2)
        module.flush()
        module.forward(1, 2)

        # Attempt another flush while the thread is still running
        with self.assertLogs(level="WARNING") as log:
            module.flush()
        self.assertIn("A plotting thread is already running", log.output[0])

    def test_switch_plotting(self):
        module = MockPlottingModule(plotting_active=False)
        module.switch_plotting(True)
        self.assertTrue(module.plotting_active)

    def test_switch_all_plotting(self):
        class MockModel(torch.nn.Module):
            def __init__(self):
                super(MockModel, self).__init__()
                self.module1 = MockPlottingModule(name="Module1")
                self.module2 = MockPlottingModule(name="Module2")
                self.module3 = torch.nn.Linear(10, 5)  # Not a PlottingModule

        model = MockModel()
        switch_all_plotting(model, True)

        self.assertTrue(model.module1.plotting_active)
        self.assertTrue(model.module2.plotting_active)

    def test_timeout_behavior(self):
        module = MockPlottingModule(plotting_active=True, timeout=0.1)
        module.forward(1, 2)
        module.flush()

        # Mock a long-running thread
        module._plot_thread.join = lambda timeout: None  # Simulate thread timeout
        with self.assertLogs(level="WARNING") as log:
            module._join_plot_thread()
        self.assertIn("Plotting thread did not finish within the timeout", log.output[0])


if __name__ == "__main__":
    unittest.main()