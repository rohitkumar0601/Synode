"""Fidelity computed in the eval env: SDMetrics quality.

Detection AUC and alpha-precision / beta-recall come from synthcity (the `metrics` stage in
the synthcity env, stl_synthcity.metrics) and are merged by stl_eval.evaluate.
"""

from __future__ import annotations

import pandas as pd
from sdmetrics.reports import QualityReport
from stl_core.metadata import TableMetadata


def quality(real: pd.DataFrame, synth: pd.DataFrame, metadata: TableMetadata):
    cols = metadata.model_columns
    md = {"tables": {"t": {"columns": {c: {"sdtype": metadata.columns[c].value} for c in cols}}}}
    report = QualityReport()
    report.generate({"t": real[cols]}, {"t": synth[cols]}, md, verbose=False)
    props = report.get_properties().set_index("Property")["Score"]
    return (float(report.get_score()), float(props["Column Shapes"]),
            float(props["Column Pair Trends"]))
