import os
import unittest
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


class TestArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.model_path = os.path.join(cls.repo_root, "indian_house_model.joblib")
        cls.viz_data_path = os.path.join(cls.repo_root, "viz_data.joblib")

    def test_model_artifact_exists_and_loads(self):
        """Confirm indian_house_model.joblib exists and loads as an estimator."""
        self.assertTrue(os.path.isfile(self.model_path), f"Missing model artifact: {self.model_path}")
        model = joblib.load(self.model_path)
        self.assertIsInstance(model, LinearRegression)
        self.assertTrue(hasattr(model, "predict"))

    def test_model_features_and_raw_prediction(self):
        """Inspect estimator feature names and verify prediction using the app's [[area]] input style."""
        model = joblib.load(self.model_path)

        # Inspect feature_names_in_
        self.assertTrue(hasattr(model, "feature_names_in_"))
        self.assertEqual(list(model.feature_names_in_), ["Area"])
        self.assertEqual(getattr(model, "n_features_in_", None), 1)

        # In app.py line 22: model.predict([[area]])[0]
        # Since model was trained with feature names, passing 2D array triggers a UserWarning.
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            prediction = model.predict([[1500]])

        # Verify prediction is numeric, non-empty, and finite
        self.assertGreater(len(prediction), 0)
        self.assertTrue(np.isfinite(prediction[0]))
        self.assertGreater(prediction[0], 0)

        # Verify and document the feature-name warning triggered by raw 2D list input
        feature_name_warnings = [
            w for w in captured_warnings
            if issubclass(w.category, UserWarning) and "feature names" in str(w.message)
        ]
        self.assertGreaterEqual(
            len(feature_name_warnings),
            1,
            "Expected feature name warning when predicting with raw 2D array without feature names",
        )

    def test_model_prediction_with_dataframe(self):
        """Separately test prediction with a DataFrame matching feature_names_in_ ('Area')."""
        model = joblib.load(self.model_path)
        df_input = pd.DataFrame({"Area": [1500]})

        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            df_prediction = model.predict(df_input)

        self.assertEqual(len(df_prediction), 1)
        self.assertTrue(np.isfinite(df_prediction[0]))
        self.assertGreater(df_prediction[0], 0)

        # With named DataFrame columns, no UserWarning regarding feature names should occur
        feature_name_warnings = [
            w for w in captured_warnings
            if issubclass(w.category, UserWarning) and "feature names" in str(w.message)
        ]
        self.assertEqual(len(feature_name_warnings), 0)

    def test_viz_data_artifact_exists_and_loads(self):
        """Confirm viz_data.joblib exists and loads as a dictionary with required keys."""
        self.assertTrue(os.path.isfile(self.viz_data_path), f"Missing viz_data artifact: {self.viz_data_path}")
        viz_data = joblib.load(self.viz_data_path)
        self.assertIsInstance(viz_data, dict)

        expected_keys = {"chennai_df", "X_test", "y_test", "y_pred", "bedroom_counts", "area_df"}
        self.assertTrue(expected_keys.issubset(viz_data.keys()))

    def test_viz_data_columns_and_structures(self):
        """Verify the specific columns and structures used by app.py."""
        viz_data = joblib.load(self.viz_data_path)

        # X_test must contain 'Area'
        self.assertIn("Area", viz_data["X_test"].columns)

        # chennai_df must contain 'Area' and 'Price'
        self.assertIn("Area", viz_data["chennai_df"].columns)
        self.assertIn("Price", viz_data["chennai_df"].columns)

        # area_df must contain 'Location' and 'House Count'
        self.assertIn("Location", viz_data["area_df"].columns)
        self.assertIn("House Count", viz_data["area_df"].columns)

        # bedroom_counts must be a non-empty Series
        self.assertIsInstance(viz_data["bedroom_counts"], pd.Series)
        self.assertGreater(len(viz_data["bedroom_counts"]), 0)

    def test_viz_data_evaluation_metrics(self):
        """Verify y_test and y_pred lengths match and r2_score computes a finite scalar."""
        viz_data = joblib.load(self.viz_data_path)

        y_test = viz_data["y_test"]
        y_pred = viz_data["y_pred"]

        self.assertEqual(len(y_test), len(y_pred))
        self.assertGreater(len(y_test), 0)

        r2 = r2_score(y_test, y_pred)
        self.assertTrue(np.isfinite(r2))
        self.assertGreater(r2, 0.0, "R2 score should be positive")


if __name__ == "__main__":
    unittest.main()
