import tempfile
import unittest
from pathlib import Path

import pandas as pd

from financeguard.explain import save_importance_plot


class ExplainabilityTests(unittest.TestCase):

    def test_importance_plot_is_created(self):
        importance = pd.DataFrame(
            {
                "feature": ["Age", "Balance"],
                "importance_mean": [0.20, 0.05],
                "importance_std": [0.02, 0.01],
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "importance.png"

            save_importance_plot(
                importance,
                output_path,
            )

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()