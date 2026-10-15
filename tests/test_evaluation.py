import unittest
import pandas as pd
import numpy as np

from financeguard.evaluation import (
    BusinessAssumptions,
    lift_at_fraction,
    select_threshold,
    threshold_table,
)


class EvaluationTests(unittest.TestCase):
    def test_threshold_economics(self):
        table = threshold_table(
            np.array([1, 0, 1, 0]),
            np.array([0.9, 0.8, 0.7, 0.1]),
            BusinessAssumptions(contact_cost=10, retained_customer_value=100, campaign_success_rate=0.5),
            thresholds=np.array([0.75]),
        )
        row = table.iloc[0]
        self.assertEqual(row["contacts"], 2)
        self.assertEqual(row["contact_rate"], 0.5)
        self.assertEqual(row["true_positives"], 1)
        self.assertEqual(row["expected_net_benefit"], 30)

    def test_perfect_ranking_has_positive_lift(self):
        lift = lift_at_fraction(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
        self.assertEqual(lift, 2.0)

    def test_threshold_respects_campaign_capacity(self):
        table = pd.DataFrame(
            {
                "threshold": [0.10, 0.20, 0.30],
                "contact_rate": [0.40, 0.20, 0.10],
                "expected_net_benefit": [100, 80, 50],
            }
        )

        unconstrained = select_threshold(table)
        constrained = select_threshold(
            table,
            max_contact_rate=0.20,
        )

        self.assertEqual(unconstrained, 0.10)
        self.assertEqual(constrained, 0.20)


if __name__ == "__main__":
    unittest.main()
