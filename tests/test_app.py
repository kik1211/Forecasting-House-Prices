import os
import unittest
import numpy as np
from streamlit.testing.v1 import AppTest


class TestStreamlitApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.app_path = os.path.join(cls.repo_root, "app.py")

    def test_app_initial_mount_and_components(self):
        """Test that the application launches without uncaught exceptions and renders expected components."""
        orig_cwd = os.getcwd()
        try:
            os.chdir(self.repo_root)
            at = AppTest.from_file(self.app_path, default_timeout=30).run()

            # The application launches without uncaught exceptions
            self.assertEqual(len(at.exception), 0, f"Uncaught exceptions during mount: {at.exception}")

            # Verify slider exists with default value of 1500
            self.assertGreater(len(at.slider), 0, "No slider found in application")
            slider = at.slider[0]
            self.assertEqual(slider.value, 1500)
            self.assertEqual(slider.label, "Select Area (sq ft)")

            # Verify R-squared metric appears with exact label used in app.py
            self.assertGreater(len(at.metric), 0, "No metric found in application")
            metric = at.metric[0]
            self.assertEqual(metric.label, "Model R-squared Score")
            metric_val = float(metric.value)
            self.assertTrue(np.isfinite(metric_val), f"Metric value is not finite: {metric.value}")
            self.assertGreater(metric_val, 0.0, "R-squared metric value should be positive")

            # Verify 4 tabs are present
            self.assertEqual(len(at.tabs), 4)

            # Verify 4 Plotly charts rendered without exceptions
            plotly_charts = at.get("plotly_chart")
            self.assertEqual(len(plotly_charts), 4)

            # Before clicking button, no success message should be shown
            self.assertEqual(len(at.success), 0)
        finally:
            os.chdir(orig_cwd)

    def test_app_prediction_button_flow(self):
        """Test clicking the prediction button executes prediction path and renders success message."""
        orig_cwd = os.getcwd()
        try:
            os.chdir(self.repo_root)
            at = AppTest.from_file(self.app_path, default_timeout=30).run()

            # Ensure button is present
            self.assertGreater(len(at.button), 0, "No button found in application")
            button = at.button[0]
            self.assertEqual(button.label, "Predict Price")

            # Click prediction button
            at = button.click().run()

            # Verify execution completed without uncaught exceptions
            self.assertEqual(len(at.exception), 0, f"Uncaught exceptions after prediction click: {at.exception}")

            # A success message is rendered with currency format and finite positive value
            self.assertGreater(len(at.success), 0, "No success message rendered after clicking Predict Price")
            success_message = at.success[0].value
            self.assertIn("Predicted Price:", success_message)
            self.assertIn("₹", success_message)

            # Extract currency numeric substring and verify finite positive value
            price_str = success_message.split("₹")[-1].replace(",", "").strip()
            price_val = float(price_str)
            self.assertTrue(np.isfinite(price_val), f"Predicted price is not finite: {price_str}")
            self.assertGreater(price_val, 0.0, "Predicted price should be positive")
        finally:
            os.chdir(orig_cwd)


if __name__ == "__main__":
    unittest.main()
