import os
import unittest
 
import pandas as pd
 
from correlation_tasks import analyze_brain_correlations
from grader_contracts.correlation_tasks import BrainDataInput
from grader_contracts.pandas_tasks import TitanicInput
from pandas_tasks import analyze_titanic
 
HERE = os.path.dirname(os.path.abspath(__file__))
TITANIC = os.path.join(HERE, "titanic.csv")
BRAIN = os.path.join(HERE, "brainsize.txt")
 
 
class TitanicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = analyze_titanic(TitanicInput(TITANIC))
        cls.df = pd.read_csv(TITANIC)
 
    def test_row_count(self):
        self.assertEqual(self.res.row_count, 891)
 
    def test_missing(self):
        m = self.res.missing_by_column
        self.assertEqual(set(m), set(self.df.columns))
        self.assertEqual(m["Age"], 177)
        self.assertEqual(m["Cabin"], 687)
        self.assertEqual(m["Embarked"], 2)
        self.assertEqual(m["Fare"], 0)
 
    def test_over_30(self):
        self.assertEqual(self.res.adults_over_30_count, 305)
 
    def test_class_keys_are_int(self):
        for d in (self.res.mean_age_by_pclass, self.res.survival_rate_by_pclass):
            self.assertEqual(sorted(d), [1, 2, 3])
            self.assertTrue(all(type(k) is int for k in d))
 
    def test_mean_age_ignores_nan(self):
        for c, v in self.res.mean_age_by_pclass.items():
            expected = self.df.loc[self.df.Pclass == c, "Age"].dropna().mean()
            self.assertAlmostEqual(v, expected)
        self.assertGreater(min(self.res.mean_age_by_pclass.values()), 20)
 
    def test_survival_rate(self):
        for c, v in self.res.survival_rate_by_pclass.items():
            self.assertTrue(0 <= v <= 1)
        self.assertAlmostEqual(self.res.survival_rate_by_pclass[1], 136 / 216)
 
    def test_highest_fares(self):
        f = self.res.highest_fares
        self.assertEqual(len(f), 5)
        self.assertEqual(f, sorted(f, reverse=True))
        self.assertEqual(f, [512.3292, 512.3292, 512.3292, 263.0, 263.0])
        self.assertTrue(all(type(x) is float for x in f))
 
 
class BrainTests(unittest.TestCase):
    FEATURES = {"FSIQ", "VIQ", "PIQ", "Weight", "Height"}
 
    @classmethod
    def setUpClass(cls):
        cls.res = analyze_brain_correlations(BrainDataInput(BRAIN))
        cls.df = pd.read_csv(BRAIN, sep="\t", na_values=["NA"])
 
    def test_counts(self):
        self.assertEqual(self.res.men_count, 20)
        self.assertEqual(self.res.women_count, 20)
 
    def test_keys_and_range(self):
        for d in (self.res.men_mri_correlation, self.res.women_mri_correlation):
            self.assertEqual(set(d), self.FEATURES)
            self.assertTrue(all(-1 <= v <= 1 for v in d.values()))
 
    def test_values_match_pandas(self):
        for gender, d in (("Male", self.res.men_mri_correlation),
                          ("Female", self.res.women_mri_correlation)):
            g = self.df[self.df.Gender == gender]
            for f, v in d.items():
                self.assertAlmostEqual(v, g[f].corr(g["MRI_Count"]))
 
    def test_weight_men_na_handled(self):
        self.assertAlmostEqual(self.res.men_mri_correlation["Weight"], -0.0769, places=3)
 
    def test_strongest_feature(self):
        self.assertEqual(self.res.strongest_mri_feature, "PIQ")
        allv = {**{("m", k): v for k, v in self.res.men_mri_correlation.items()},
                **{("w", k): v for k, v in self.res.women_mri_correlation.items()}}
        best = max(allv, key=lambda k: abs(allv[k]))
        self.assertEqual(best[1], self.res.strongest_mri_feature)
 
 
if __name__ == "__main__":
    unittest.main()