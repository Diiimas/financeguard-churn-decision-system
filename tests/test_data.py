import tempfile
import unittest
from pathlib import Path

import pandas as pd

from financeguard.data import FEATURE_COLUMNS, load_dataset


class LoadDatasetTests(unittest.TestCase):
    def test_removes_identifiers_and_preserves_expected_features(self):
        row = {column: 1 for column in FEATURE_COLUMNS}
        row.update({"Geography": "France", "Gender": "Female", "Exited": 0, "CustomerId": 7})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            pd.DataFrame([row]).to_csv(path, index=False)
            features, target = load_dataset(path)
        self.assertEqual(tuple(features.columns), FEATURE_COLUMNS)
        self.assertNotIn("CustomerId", features.columns)
        self.assertEqual(target.tolist(), [0])

    def test_rejects_missing_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            pd.DataFrame({"Exited": [0]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "Missing required columns"):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
