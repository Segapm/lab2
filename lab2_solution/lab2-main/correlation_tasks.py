"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
TARGET = "MRI_Count"


def _mri_correlations(group: pd.DataFrame) -> dict[str, float]:
    # Series.corr считает Пирсона по парам без пропусков
    return {f: float(group[f].corr(group[TARGET], method="pearson")) for f in FEATURES}


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = pd.read_csv(data.csv_path, sep="\t", na_values=["NA"])
    df["Gender"] = df["Gender"].astype(str).str.strip().str.lower()

    men = df[df["Gender"] == "male"]
    women = df[df["Gender"] == "female"]

    men_corr = _mri_correlations(men)
    women_corr = _mri_correlations(women)

    candidates = [(abs(v), f) for corr in (men_corr, women_corr) for f, v in corr.items()
                  if pd.notna(v)]
    strongest = max(candidates)[1]

    return BrainCorrelationSummary(
        men_count=int(len(men)),
        women_count=int(len(women)),
        women_mri_correlation=women_corr,
        men_mri_correlation=men_corr,
        strongest_mri_feature=strongest,
    )
