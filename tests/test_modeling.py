import unittest

import pandas as pd

from financeguard.modeling import build_preprocessor


class PreprocessorTests(unittest.TestCase):

    def test_one_hot_encoding_uses_reference_categories(self):
        data = pd.DataFrame(
            {
                "CreditScore": [600, 700, 650],
                "Age": [30, 40, 50],
                "Tenure": [1, 5, 10],
                "Balance": [0, 50000, 100000],
                "NumOfProducts": [1, 2, 3],
                "EstimatedSalary": [50000, 70000, 90000],
                "Geography": ["France", "Germany", "Spain"],
                "Gender": ["Female", "Male", "Female"],
                "HasCrCard": [0, 1, 1],
                "IsActiveMember": [0, 1, 0],
            }
        )

        preprocessor = build_preprocessor()
        transformed = preprocessor.fit_transform(data)
        feature_names = preprocessor.get_feature_names_out()

        self.assertEqual(transformed.shape[1], 11)
        self.assertNotIn("Gender_Female", feature_names)
        self.assertNotIn("Geography_France", feature_names)


if __name__ == "__main__":
    unittest.main()
